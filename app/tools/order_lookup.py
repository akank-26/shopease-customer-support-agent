
import json
import re
from pathlib import Path

from langchain.tools import tool


ROOT = Path(__file__).resolve().parents[2]
ORDERS_FILE = ROOT / "data" / "orders.json"


@tool
def get_order_status(order_id: str) -> dict:
    """Look up the current status of a ShopEase order by order ID."""

    order_id = order_id.strip().upper()

    if not re.fullmatch(r"ORD-\d{4}", order_id):
        return {
            "error": "Invalid order ID format",
            "expected_format": "ORD-1234",
        }

    with ORDERS_FILE.open(encoding="utf-8") as file:
        orders = json.load(file)

    order = orders.get(order_id)

    if order is None:
        return {
            "error": "Order not found",
            "order_id": order_id,
        }

    return {
        "order_id": order_id,
        **order,
    }
