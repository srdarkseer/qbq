import json
import textwrap

import pytest

from cubic.cli import main
from cubic.compiler import TargetError
from cubic.compiler.adapter import external_environment, load_personas
from cubic.graph import ActorPersona
from cubic.remediate import LLMPatcher, PatchPrompt, RulePatcher, remediate
from cubic.remediate.patcher import unified_diff

DEMO_APP = textwrap.dedent('''\
    """A third-party shop API: not a Bug Gym app, no feature flags, real-world-style bugs."""

    from typing import Annotated, Any

    from fastapi import Depends, FastAPI, Header, HTTPException
    from pydantic import BaseModel


    class OrderIn(BaseModel):
        sku: str
        quantity: int


    def create_app():
        tokens = {"tok_ann": "ann", "tok_ben": "ben"}
        orders = {
            "o_ann": {"id": "o_ann", "owner_id": "ann", "sku": "mug", "quantity": 1},
            "o_ben": {"id": "o_ben", "owner_id": "ben", "sku": "pen", "quantity": 3},
        }
        app = FastAPI(title="Demo Shop")

        def current_user(authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(401, "missing bearer token")
            user_id = tokens.get(authorization.removeprefix("Bearer "))
            if user_id is None:
                raise HTTPException(401, "invalid token")
            return {"id": user_id}

        User = Annotated[dict[str, Any], Depends(current_user)]

        @app.get("/orders/{order_id}")
        async def read_order(order_id: str, user: User) -> dict[str, Any]:
            order = orders.get(order_id)
            if order is None:
                raise HTTPException(404, "order not found")
            return order

        @app.post("/orders", status_code=201)
        async def create_order(body: OrderIn, user: User) -> dict[str, Any]:
            order_id = f"o_{len(orders) + 1}"
            orders[order_id] = {"id": order_id, "owner_id": user["id"], "sku": body.sku,
                                "quantity": body.quantity}
            return orders[order_id]

        return app
''')

PERSONAS = [
    ActorPersona(id="anonymous", display_name="Anonymous", role="ANONYMOUS"),
    ActorPersona(id="ann", display_name="Ann", role="USER", auth_token="tok_ann",
                 owned_resources={"orders": frozenset({"o_ann"})}),
    ActorPersona(id="ben", display_name="Ben", role="USER", auth_token="tok_ben",
                 owned_resources={"orders": frozenset({"o_ben"})}),
]


@pytest.fixture
def demo(tmp_path, monkeypatch):
    (tmp_path / "demo_shop.py").write_text(DEMO_APP)
    personas = tmp_path / "personas.json"
    personas.write_text(json.dumps([p.model_dump(mode="json") for p in PERSONAS]))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path, personas


class _OwnershipFixModel:
    """Stands in for a code model: proposes the missing ownership check as a unified diff."""

    def complete(self, prompt: PatchPrompt) -> str:
        assert "tok_" not in prompt.render()
        source = DEMO_APP
        fixed = source.replace(
            '            raise HTTPException(404, "order not found")\n        return order\n',
            '            raise HTTPException(404, "order not found")\n'
            '        if order["owner_id"] != user["id"]:\n'
            '            raise HTTPException(404, "order not found")\n'
            '        return order\n',
        )
        assert fixed != source
        return "```diff\n" + unified_diff("demo_shop.py", source, fixed) + "```"


def test_external_environment_compiles_and_grounds(demo):
    env = external_environment("demo_shop:create_app", PERSONAS)
    ids = {i.id for i in env.graph.invariants}
    assert {"inv.tenant.read_order", "inv.input.create_order.quantity", "inv.auth.read_order"} <= ids
    assert env.bugs == () and env.app_source.accepts_flags is False
    assert env.graph.actor("ann").observed("orders", "o_ann")["quantity"] == 1
    assert [v.name for v in env.variants()] == ["patched", "vulnerable"]


def test_external_remediation_rule_and_llm_patchers(demo):
    tmp_path, _ = demo
    env = external_environment("demo_shop:create_app", PERSONAS)
    report = remediate(env, runs=3, out_dir=tmp_path / "regression",
                       patchers=[RulePatcher(), LLMPatcher(_OwnershipFixModel())])
    by_inv = {f.invariant_id: f for f in report.findings}
    assert set(by_inv) == {"inv.tenant.read_order", "inv.input.create_order.quantity"}
    # the input bug has a deterministic schema fix; the deleted guard needs the model
    assert by_inv["inv.input.create_order.quantity"].verified_patch.rule == "schema_lower_bound"
    assert by_inv["inv.tenant.read_order"].verified_patch.patcher == "llm"
    assert report.patch_pass_rate == 1.0
    for f in report.findings:
        assert f.regression_test.fails_on_unpatched and f.regression_test.passes_on_patched
    exported = (tmp_path / "regression" / "test_regression_inv_tenant_read_order.py").read_text()
    assert "from demo_shop import create_app" in exported


def test_rule_patcher_alone_reports_unfixed_deleted_guard(demo):
    env = external_environment("demo_shop:create_app", PERSONAS)
    report = remediate(env, runs=1)
    tenant = next(f for f in report.findings if f.invariant_id == "inv.tenant.read_order")
    assert tenant.patches == [] and not tenant.verified


def test_cli_compile_and_remediate_with_app_flag(demo, capsys):
    tmp_path, personas = demo
    assert main(["compile", "--app", "demo_shop:create_app", "--personas", str(personas),
                 "--out", str(tmp_path / "graph.json")]) == 0
    assert "compiled demo-shop" in capsys.readouterr().out
    code = main(["remediate", "--app", "demo_shop:create_app", "--personas", str(personas), "--runs", "1",
                 "--out-dir", str(tmp_path / "rt"), "--require-verified"])
    out = capsys.readouterr()
    assert code == 1  # the deleted ownership check has no deterministic rule fix
    assert "1/2 findings have a verified patch" in out.out and "GATE FAILED" in out.err


def test_adapter_rejects_bad_targets(demo):
    with pytest.raises(TargetError, match="importable"):
        external_environment("demo_shop.py:create_app")
    with pytest.raises(TargetError, match="cannot import"):
        external_environment("no_such_module:create_app")
    with pytest.raises(TargetError, match="factory"):
        external_environment("demo_shop:OrderIn_missing")
    tmp_path, personas = demo
    assert [p.id for p in load_personas(personas)] == ["anonymous", "ann", "ben"]
