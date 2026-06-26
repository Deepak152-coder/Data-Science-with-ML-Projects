"""
app.py
Single-file Streamlit frontend for the FastAPI Product Management API.

Run with:
    streamlit run app.py

Requires:
    pip install streamlit requests pandas
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 8  # seconds

st.set_page_config(
    page_title="Product Management Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------


def _handle_response(resp: requests.Response) -> Tuple[bool, Any]:
    """Parse a requests.Response into (success, data_or_error_message)."""
    try:
        body = resp.json()
    except ValueError:
        body = resp.text

    if resp.ok:
        return True, body

    if isinstance(body, dict) and "detail" in body:
        error_msg = body["detail"]
    else:
        error_msg = body or f"Request failed with status {resp.status_code}"
    return False, error_msg


def _request(method: str, path: str, **kwargs) -> Tuple[bool, Any]:
    """Centralized request wrapper with friendly exception handling."""
    url = f"{BASE_URL}{path}"
    try:
        resp = requests.request(method, url, timeout=TIMEOUT, **kwargs)
        return _handle_response(resp)
    except requests.exceptions.ConnectionError:
        return False, (
            "Could not connect to the API. Make sure the FastAPI backend is "
            f"running at {BASE_URL}."
        )
    except requests.exceptions.Timeout:
        return False, "The request timed out. Please try again."
    except requests.exceptions.RequestException as exc:
        return False, f"An unexpected error occurred: {exc}"


def get_products(
    limit: int = 10,
    offset: int = 0,
    name: Optional[str] = None,
    sort_by_price: bool = False,
    order: str = "asc",
) -> Tuple[bool, Any]:
    """Fetch a list of products with optional filtering/sorting."""
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    if name:
        params["name"] = name
    if sort_by_price:
        params["sort_by_price"] = True
        params["order"] = order
    return _request("GET", "/products", params=params)


def get_product(product_id: str) -> Tuple[bool, Any]:
    """Fetch a single product by its ID."""
    return _request("GET", f"/products/{product_id}")


def create_product(payload: Dict[str, Any]) -> Tuple[bool, Any]:
    """Create a new product."""
    return _request("POST", "/products", json=payload)


def update_product(product_id: str, payload: Dict[str, Any]) -> Tuple[bool, Any]:
    """Update an existing product."""
    return _request("PUT", f"/products/{product_id}", json=payload)


def delete_product(product_id: str) -> Tuple[bool, Any]:
    """Delete a product by its ID."""
    return _request("DELETE", f"/products/{product_id}")


def normalize_product_list(data: Any) -> List[Dict[str, Any]]:
    """Normalize various response shapes into a list of product dicts."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("products", "items", "data", "results"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def render_page_header(emoji: str, title: str, subtitle: str = "") -> None:
    """Render a consistent page header with emoji, title and optional subtitle."""
    st.markdown(f"## {emoji} {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()


# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"
if "products_data" not in st.session_state:
    st.session_state.products_data = None
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False
if "delete_target" not in st.session_state:
    st.session_state.delete_target = ""


def go_to(page_name: str) -> None:
    st.session_state.page = page_name


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
PAGES = [
    "🏠 Home",
    "📦 All Products",
    "🔍 Search Product",
    "➕ Create Product",
    "✏️ Update Product",
    "🗑️ Delete Product",
]

with st.sidebar:
    st.markdown("# 🛒 Product Admin")
    st.caption(f"Connected to `{BASE_URL}`")
    st.divider()
    st.session_state.page = st.radio(
        "Navigate", PAGES, index=PAGES.index(st.session_state.page), label_visibility="collapsed"
    )

page = st.session_state.page

# --------------------------------------------------------------------------
# 🏠 HOME PAGE
# --------------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown("# 🛒 Product Management Dashboard")
    st.caption("A clean, fast admin console for your FastAPI Product Management API.")
    st.divider()

    with st.container(border=True):
        st.markdown("### 👋 Welcome")
        st.write(
            "This dashboard gives you a complete, no-code interface to manage products "
            "exposed by your FastAPI backend — browse, search, create, update, and delete "
            "products without ever touching raw JSON or Postman. All requests are sent "
            f"live to `{BASE_URL}`."
        )

    st.write("")
    st.markdown("### 🚀 Quick Navigation")
    st.write("")

    nav_cards = [
        ("📦", "All Products", "Browse, filter & sort the full catalog"),
        ("🔍", "Search Product", "Look up a single product by its ID"),
        ("➕", "Create Product", "Add a brand new product"),
        ("✏️", "Update Product", "Edit an existing product's details"),
        ("🗑️", "Delete Product", "Remove a product permanently"),
    ]

    row1 = st.columns(3)
    row2 = st.columns(2)
    slots = row1 + row2

    for slot, (emoji, title, desc) in zip(slots, nav_cards):
        with slot:
            with st.container(border=True):
                st.markdown(f"#### {emoji} {title}")
                st.caption(desc)
                target = f"{emoji} {title}"
                st.button(
                    f"Go to {title}",
                    key=f"nav_{title}",
                    use_container_width=True,
                    on_click=go_to,
                    args=(target,),
                )

    st.write("")
    st.divider()
    with st.container(border=True):
        st.markdown("#### ℹ️ About this project")
        st.write(
            "Built with **Streamlit** and **Requests** only — no Flask, no React, no "
            "HTML/JS templates. Every page talks directly to your FastAPI backend's "
            "`/products` endpoints and presents the results as a polished native UI."
        )
        cols = st.columns(4)
        cols[0].metric("Frontend", "Streamlit")
        cols[1].metric("HTTP Client", "Requests")
        cols[2].metric("Backend", "FastAPI")
        cols[3].metric("Pages", "5")

# --------------------------------------------------------------------------
# 📦 ALL PRODUCTS PAGE
# --------------------------------------------------------------------------
elif page == "📦 All Products":
    render_page_header("📦", "All Products", "Browse and filter your full product catalog.")

    with st.container(border=True):
        st.markdown("#### ⚙️ Filters")
        c1, c2 = st.columns(2)
        with c1:
            limit = st.slider("Number of Products", min_value=1, max_value=100, value=10)
        with c2:
            offset = st.number_input("Offset", min_value=0, value=0, step=1)

        c3, c4, c5 = st.columns(3)
        with c3:
            name = st.text_input("Search by name", placeholder="e.g. Wireless Mouse")
        with c4:
            sort_by_price = st.checkbox("Sort by price")
        with c5:
            order = st.selectbox("Order", ["asc", "desc"], disabled=not sort_by_price)
            order_label = "Ascending" if order == "asc" else "Descending"
            st.caption(f"Order: {order_label}" if sort_by_price else "Order: —")

        load_clicked = st.button("🔄 Load Products", type="primary", use_container_width=True)

    if load_clicked:
        with st.spinner("Fetching products..."):
            success, data = get_products(
                limit=int(limit),
                offset=int(offset),
                name=name.strip() or None,
                sort_by_price=sort_by_price,
                order=order,
            )
        if success:
            st.session_state.products_data = data
        else:
            st.session_state.products_data = None
            st.error(f"❌ Failed to load products: {data}")

    st.write("")

    if st.session_state.products_data is not None:
        products = normalize_product_list(st.session_state.products_data)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Products", len(products))
        m2.metric("Returned Products", len(products))
        m3.metric("Offset", int(offset))

        st.write("")
        with st.container(border=True):
            if products:
                df = pd.DataFrame(products)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No products found for the given filters.")
    else:
        st.info("Set your filters above and click **Load Products** to get started.")

# --------------------------------------------------------------------------
# 🔍 SEARCH PRODUCT PAGE
# --------------------------------------------------------------------------
elif page == "🔍 Search Product":
    render_page_header("🔍", "Search Product", "Look up a single product by its ID.")

    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        with c1:
            product_id = st.text_input("Product ID", placeholder="e.g. PROD-1023")
        with c2:
            st.write("")
            st.write("")
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

    if search_clicked:
        if not product_id.strip():
            st.warning("⚠️ Please enter a Product ID.")
        else:
            with st.spinner("Searching..."):
                success, data = get_product(product_id.strip())

            st.write("")
            if success and data:
                st.success("✅ Product found!")
                with st.container(border=True):
                    st.markdown("#### 📋 Product Details")
                    if isinstance(data, dict):
                        cols = st.columns(min(len(data), 4) or 1)
                        for i, (key, value) in enumerate(data.items()):
                            cols[i % len(cols)].metric(str(key).replace("_", " ").title(), str(value))
                        st.write("")
                        df = pd.DataFrame([data])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.write(data)
            else:
                st.error(f"❌ Product not found: {data}")

# --------------------------------------------------------------------------
# ➕ CREATE PRODUCT PAGE
# --------------------------------------------------------------------------
elif page == "➕ Create Product":
    render_page_header("➕", "Create Product", "Add a brand new product to the catalog.")

    with st.container(border=True):
        st.markdown("#### 🆕 Product Details")
        with st.form("create_product_form", clear_on_submit=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                product_id = st.text_input("Product ID *", placeholder="e.g. PROD-1023")
            with c2:
                sku = st.text_input("SKU *", placeholder="e.g. SKU-998877")
            with c3:
                product_name = st.text_input("Product Name *", placeholder="e.g. Wireless Mouse")

            submitted = st.form_submit_button(
                "➕ Create Product", type="primary", use_container_width=True
            )

    if submitted:
        if not product_id.strip() or not sku.strip() or not product_name.strip():
            st.warning("⚠️ Please fill in all required fields.")
        else:
            payload = {
                "product_id": product_id.strip(),
                "sku": sku.strip(),
                "name": product_name.strip(),
            }
            with st.spinner("Creating product..."):
                success, data = create_product(payload)

            st.write("")
            if success:
                st.success("✅ Product created successfully!")
                st.balloons()
                with st.container(border=True):
                    st.markdown("#### 📦 Created Product")
                    if isinstance(data, dict):
                        df = pd.DataFrame([data])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.write(data)
            else:
                st.error(f"❌ Failed to create product: {data}")

# --------------------------------------------------------------------------
# ✏️ UPDATE PRODUCT PAGE
# --------------------------------------------------------------------------
elif page == "✏️ Update Product":
    render_page_header("✏️", "Update Product", "Edit an existing product's details.")

    with st.container(border=True):
        st.markdown("#### 🔎 Existing Product")
        existing_id = st.text_input("Existing Product ID *", placeholder="e.g. PROD-1023")

    st.write("")

    with st.container(border=True):
        st.markdown("#### ✏️ New Details")
        with st.form("update_product_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                new_id = st.text_input("New Product ID *", placeholder="e.g. PROD-1023")
            with c2:
                new_sku = st.text_input("New SKU *", placeholder="e.g. SKU-998877")
            with c3:
                new_name = st.text_input("New Product Name *", placeholder="e.g. Wireless Mouse Pro")

            submitted = st.form_submit_button(
                "💾 Update Product", type="primary", use_container_width=True
            )

    if submitted:
        if not existing_id.strip():
            st.warning("⚠️ Please enter the existing Product ID.")
        elif not new_id.strip() or not new_sku.strip() or not new_name.strip():
            st.warning("⚠️ Please fill in all required new-detail fields.")
        else:
            payload = {
                "product_id": new_id.strip(),
                "sku": new_sku.strip(),
                "name": new_name.strip(),
            }
            with st.spinner("Updating product..."):
                success, data = update_product(existing_id.strip(), payload)

            st.write("")
            if success:
                st.success("✅ Product updated successfully!")
                with st.container(border=True):
                    st.markdown("#### 📦 Updated Product")
                    if isinstance(data, dict):
                        df = pd.DataFrame([data])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.write(data)
            else:
                st.error(f"❌ Failed to update product: {data}")

# --------------------------------------------------------------------------
# 🗑️ DELETE PRODUCT PAGE
# --------------------------------------------------------------------------
elif page == "🗑️ Delete Product":
    render_page_header("🗑️", "Delete Product", "Remove a product permanently from the catalog.")

    with st.container(border=True):
        product_id = st.text_input("Product ID", placeholder="e.g. PROD-1023")

        if product_id.strip() != st.session_state.delete_target:
            st.session_state.confirm_delete = False
            st.session_state.delete_target = product_id.strip()

        danger_clicked = st.button(
            "🗑️ DELETE PRODUCT", type="primary", use_container_width=True
        )

    if danger_clicked:
        if not product_id.strip():
            st.warning("⚠️ Please enter a Product ID.")
        else:
            st.session_state.confirm_delete = True

    if st.session_state.confirm_delete:
        st.write("")
        with st.container(border=True):
            st.warning(
                f"⚠️ Are you sure you want to permanently delete product "
                f"**{st.session_state.delete_target}**? This action cannot be undone."
            )
            c1, c2 = st.columns(2)
            with c1:
                confirmed = st.button("✅ Yes, delete it", use_container_width=True)
            with c2:
                cancelled = st.button("✋ Cancel", use_container_width=True)

        if cancelled:
            st.session_state.confirm_delete = False
            st.info("Deletion cancelled.")

        if confirmed:
            target_id = st.session_state.delete_target

            # Try to fetch product details before deletion, in case DELETE
            # doesn't return the deleted object.
            _, pre_data = get_product(target_id)

            with st.spinner("Deleting product..."):
                success, data = delete_product(target_id)

            st.session_state.confirm_delete = False

            st.write("")
            if success:
                st.success(f"✅ Product **{target_id}** was deleted successfully.")
                display_data = data if isinstance(data, dict) and data else pre_data
                with st.container(border=True):
                    st.markdown("#### 🗑️ Deleted Product")
                    if isinstance(display_data, dict):
                        df = pd.DataFrame([display_data])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.write(display_data)
            else:
                st.error(f"❌ Failed to delete product: {data}")