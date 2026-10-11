import sqlite3
from langgraph.store.sqlite import SqliteStore


def test_put(store):
    # 添加数据
    namespace1 = ("user_preferences", 'user_01')
    # 用户信息
    store.put(namespace1, key='profile', value={'name': '张三', 'age': 108})
    # 用户饮食偏好
    store.put(namespace1, key='food', value={
        "likes": ["烧烤", "酱鸭", "排骨"],    # 喜欢吃的
        "dislikes": ["香菜", "折耳根"],       # 不喜欢的
        "restrictions": ["鱼干"]             # 忌口
    })

    namespace2 = ("user_preferences", 'user_02')
    store.put(namespace2, key='profile', value={'name': '李四', 'age': 98})
    print(f'{"-"*50} test_put {"-"*50}')


def test_list_namespaces(store):
    # 查询所有命名空间
    print(f'{"-"*50} list_namespaces {"-"*50}')
    namespace_list = store.list_namespaces()
    print(namespace_list)


def test_search(store, namespace, filter=None):
    # 查询namespace中的所有数据
    print(f'\n{"-" * 50} test_search {"-" * 50}')

    if filter: # 条件过滤
        items = store.search(namespace, filter=filter, limit=100)
    else:     # 模糊匹配
        items = store.search(namespace, limit=100)

    for item in items:
        print(item)


def test_get(store, namespace, key):
    # 获取某个namespace下的某个key
    print(f'\n{"-" * 50} test_get {namespace}, {key} {"-" * 50}')
    result = store.get(namespace, key=key)
    print(result)


def test_delete(store, namespace, key):
    # 删除namespace中的某个key
    store.delete(namespace, key=key)
    print(f'\n{"-" * 50} test_delete {namespace}, {key} {"-" * 50}')


if __name__ == '__main__':
    conn = sqlite3.connect("./sqlite/agent_store.db", check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn)
    store.setup()  # 同样需要调用 setup() 建表

    test_put(store)
    test_list_namespaces(store)

    # namespace可以只写前缀，这样相当于查询所有数据
    test_search(store, namespace=('user_preferences',))

    # filter：按条件过滤，查询所有value中包含age属性，并且值等于108的数据
    # test_search(store, namespace=('user_preferences',), filter={'age': 108})

    # namespace写完整的，查询user_01的所有数据
    # test_search(store, namespace=('user_preferences', 'user_01'))

    # test_get(store, namespace=('user_preferences', 'user_01'), key='food')

    # test_delete(store, namespace=('user_preferences', 'user_01'), key='food')
    # test_search(store, namespace=('user_preferences',))