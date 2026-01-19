from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,

    connections
)

# Connect to Local Database (hosted on a Docker Compose)
connections.connect(host='localhost', port='19530')

# Declare FieldSchemas
fields = [
    FieldSchema(
        name='id',
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True
    ),
    FieldSchema(
        name='embedding',
        dtype=DataType.FLOAT_VECTOR,
        dim=3072,
    ),
    FieldSchema(
        name='text',
        dtype=DataType.VARCHAR,
        max_length=512
    )
]

# Declare CollectionSchema
schema = CollectionSchema(
    fields=fields,
    description="collection de teste"
)

# Create collection
collection = Collection(
    name="gemini_collection",
    schema=schema
)