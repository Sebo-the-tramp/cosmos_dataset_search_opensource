from pymilvus import DataType, MilvusClient


URI = "http://localhost:19530"
TOKEN = "root:Milvus"
COLLECTION_NAME = "cosmos_cds_test_00"
VECTOR_FIELD = "embedding"
VECTOR_DIM = 768
METRIC_TYPE = "COSINE"
VIDEO_PATH_MAX_LENGTH = 512
CHUNK_MAX_LENGTH = 128


def main() -> None:
    client = MilvusClient(uri=URI, token=TOKEN)

    if client.has_collection(COLLECTION_NAME):
        answer = input(f"Drop existing collection '{COLLECTION_NAME}'? [y/N] ")
        assert answer.lower() == "y"
        client.drop_collection(COLLECTION_NAME)

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="video_path", datatype=DataType.VARCHAR, max_length=VIDEO_PATH_MAX_LENGTH)
    schema.add_field(field_name="chunk", datatype=DataType.VARCHAR, max_length=CHUNK_MAX_LENGTH)
    schema.add_field(field_name=VECTOR_FIELD, datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=VECTOR_FIELD,
        index_type="AUTOINDEX",
        metric_type=METRIC_TYPE,
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

    print(client.describe_collection(COLLECTION_NAME))
    print(client.get_load_state(COLLECTION_NAME))


if __name__ == "__main__":
    main()
