"""Pydantic v2 request / response models."""

from typing import Annotated

from pydantic import BaseModel, Field


class OrderCreate(BaseModel):
    product_id: str
    quantity: Annotated[int, Field(gt=0, le=100)]
    shipping_address: str = Field(min_length=1)


class OrderOut(BaseModel):
    id: str
    customer_id: str
    product_id: str
    quantity: int
    total_cents: int
    status: str


class InvoiceOut(BaseModel):
    order_id: str
    customer_email: str
    shipping_address: str
    total_cents: int


class RefundIn(BaseModel):
    reason: str = Field(min_length=3)
