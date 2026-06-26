from typing import Annotated

from fastapi import FastAPI, Query, HTTPException, Path
from pydantic import BaseModel, Field

from service.products import get_all_products

app = FastAPI()

# Load products once
products = get_all_products()


# =====================================================
# ROOT
# =====================================================
@app.get("/")
def root():
    return {"message": "Welcome to FastAPI."}


# =====================================================
# GET ALL PRODUCTS
# =====================================================
@app.get("/products")
def list_products(
    name: str = Query(
        default=None,
        min_length=1,
        max_length=50,
        description="Search product by name (case insensitive)"
    ),
    sort_by_price: bool = Query(
        default=False,
        description="Sort products by price"
    ),
    order: str = Query(
        default="asc",
        description="Sort order: asc or desc"
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of products"
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of products to skip"
    )
):
    filtered_products = products

    # Search
    if name:
        keyword = name.strip().lower()

        filtered_products = [
            product
            for product in filtered_products
            if keyword in product.get("name", "").lower()
        ]

        if not filtered_products:
            raise HTTPException(
                status_code=404,
                detail=f"No product found matching '{name}'"
            )

    # Sort
    if sort_by_price:

        if order.lower() not in ["asc", "desc"]:
            raise HTTPException(
                status_code=400,
                detail="Order must be either 'asc' or 'desc'"
            )

        filtered_products = sorted(
            filtered_products,
            key=lambda product: product.get("price", 0),
            reverse=(order.lower() == "desc")
        )

    total = len(filtered_products)

    # Pagination
    filtered_products = filtered_products[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "returned": len(filtered_products),
        "items": filtered_products
    }


# =====================================================
# GET PRODUCT BY ID
# =====================================================
@app.get("/products/{product_id}")
def get_product_by_id(
    product_id: str = Path(
        ...,
        min_length=1,
        max_length=36,
        title="Product ID",
        description="Unique product id",
        examples=["10"]
    )
):
    for product in products:
        if str(product["id"]) == product_id:
            return product

    raise HTTPException(
        status_code=404,
        detail=f"Product with id '{product_id}' not found"
    )


# =====================================================
# PRODUCT MODEL
# =====================================================
class Product(BaseModel):
    id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=36,
            title="Product ID",
            examples=["101"]
        )
    ]

    sku: Annotated[
        str,
        Field(
            min_length=6,
            max_length=30,
            title="SKU",
            description="Stock Keeping Unit",
            examples=["SKU101"]
        )
    ]

    name: Annotated[
        str,
        Field(
            min_length=3,
            max_length=100,
            title="Product Name",
            examples=["Gaming Mouse"]
        )
    ]


# =====================================================
# CREATE PRODUCT
# =====================================================
@app.post("/products", status_code=201)
def create_product(product: Product):

    # Check duplicate ID
    for item in products:
        if str(item["id"]) == product.id:
            raise HTTPException(
                status_code=400,
                detail=f"Product with id '{product.id}' already exists"
            )

    products.append(product.model_dump())

    return {
        "message": "Product created successfully",
        "product": product
    }


# =====================================================
# UPDATE PRODUCT
# =====================================================
@app.put("/products/{product_id}")
def update_product(
    product_id: str,
    updated_product: Product
):
    for index, product in enumerate(products):

        if str(product["id"]) == product_id:

            products[index] = updated_product.model_dump()

            return {
                "message": "Product updated successfully",
                "product": updated_product
            }

    raise HTTPException(
        status_code=404,
        detail=f"Product with id '{product_id}' not found"
    )


# =====================================================
# PARTIAL UPDATE PRODUCT
# =====================================================
class ProductUpdate(BaseModel):
    sku: Annotated[
        str | None,
        Field(
            min_length=6,
            max_length=30,
            examples=["SKU999"]
        )
    ] = None

    name: Annotated[
        str | None,
        Field(
            min_length=3,
            max_length=100,
            examples=["Wireless Mouse"]
        )
    ] = None


@app.patch("/products/{product_id}")
def partial_update_product(
    product_id: str,
    product: ProductUpdate
):
    for item in products:

        if str(item["id"]) == product_id:

            update_data = product.model_dump(exclude_unset=True)

            item.update(update_data)

            return {
                "message": "Product updated successfully",
                "product": item
            }

    raise HTTPException(
        status_code=404,
        detail=f"Product with id '{product_id}' not found"
    )


# =====================================================
# DELETE PRODUCT
# =====================================================
@app.delete("/products/{product_id}")
def delete_product(product_id: str):

    for index, product in enumerate(products):

        if str(product["id"]) == product_id:

            deleted_product = products.pop(index)

            return {
                "message": "Product deleted successfully",
                "product": deleted_product
            }

    raise HTTPException(
        status_code=404,
        detail=f"Product with id '{product_id}' not found"
    )