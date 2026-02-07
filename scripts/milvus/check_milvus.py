from pymilvus import connections, utility, Collection

def check_milvus_status(host='localhost', port='19530'):
    try:
        # 1. Milvus 서버 연결
        print(f"Connecting to Milvus at {host}:{port}...")
        connections.connect(alias="default", host=host, port=port)
        
        # 2. 모든 컬렉션 목록 가져오기
        collections = utility.list_collections()
        
        if not collections:
            print("⚠️ Milvus 서버에 생성된 컬렉션이 하나도 없습니다. (비어있음)")
            return

        print(f"\n✅ 발견된 컬렉션 수: {len(collections)}")
        print("-" * 50)
        print(f"{'Collection Name':<30} | {'Entity Count':<15}")
        print("-" * 50)

        for name in collections:
            # 컬렉션 객체 생성
            col = Collection(name)
            # 최신 상태를 반영하기 위해 flush (데이터가 막 들어온 경우 필요)
            # col.flush() 
            
            # 데이터 개수 출력
            print(f"{name:<30} | {col.num_entities:<15}")
            
        print("-" * 50)

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
    finally:
        connections.disconnect("default")

if __name__ == "__main__":
    # 여기에 Milvus 서버의 외부 IP를 입력하세요.
    # VM 내부에서 실행한다면 'loc
    # alhost' 그대로 두시면 됩니다.
    MILVUS_HOST = '34.158.221.160' 
    check_milvus_status(host=MILVUS_HOST)