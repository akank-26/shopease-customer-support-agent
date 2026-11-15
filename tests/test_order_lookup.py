
from app.tools.order_lookup import get_order_status


def test_existing_order():
    result = get_order_status.invoke({
        "order_id": "ORD-1024"
    })

    assert result["status"] == "Shipped"
    assert result["order_id"] == "ORD-1024"


def test_unknown_order():
    result = get_order_status.invoke({
        "order_id": "ORD-9999"
    })

    assert result["error"] == "Order not found"


def test_invalid_order_id():
    result = get_order_status.invoke({
        "order_id": "invalid"
    })

    assert result["error"] == "Invalid order ID format"


def test_order_id_normalization():
    result = get_order_status.invoke({
        "order_id": " ord-1025 "
    })

    assert result["status"] == "Delivered"
