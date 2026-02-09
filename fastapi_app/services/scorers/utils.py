import os


def use_milvus() -> bool:
    flag = os.getenv("USE_MILVUS")
    if flag is None:
        return bool(os.getenv("MILVUS_HOST"))
    return flag.strip().lower() in {"1", "true", "yes"}
