import os
import sqlite3


# 1. 客户基本信息与健康画像表 (customers)
customers = '''
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(32) PRIMARY KEY,      -- 客户唯一ID
    name VARCHAR(50) NOT NULL,                -- 客户姓名
    gender VARCHAR(10) CHECK(gender IN ('男', '女')), -- 性别
    age INTEGER NOT NULL,                     -- 年龄
    city VARCHAR(50) NOT NULL,                -- 所属城市 (用于华东/华北等区域统计)
    driving_years INTEGER DEFAULT 0           -- 驾龄 (车险风控核心字段，健康险用户为0)
);
'''

# 插入 products 表数据 (2条：1个医疗险，1个车险)
customers_data = '''INSERT INTO customers (customer_id, name, gender, age, city, driving_years) VALUES
('C001', '张伟', '男', 35, '上海', 10),
('C002', '李娜', '女', 28, '北京', 5),
('C003', '王强', '男', 45, '杭州', 20),
('C004', '赵敏', '女', 52, '南京', 0),
('C005', '孙杰', '男', 22, '济南', 1),
('C006', '周涛', '男', 30, '上海', 8),
('C007', '吴静', '女', 41, '天津', 0),
('C008', '郑浩', '男', 38, '广州', 12),
('C009', '陈露', '女', 25, '深圳', 2),
('C010', '褚健', '男', 55, '成都', 25),
('C011', '卫燕', '女', 33, '北京', 0),
('C012', '蒋亮', '男', 29, '上海', 6),
('C013', '沈霞', '女', 48, '杭州', 0),
('C014', '韩峰', '男', 31, '武汉', 7),
('C015', '杨梅', '女', 27, '西安', 0),
('C016', '朱宇', '男', 40, '重庆', 15),
('C017', '秦瑶', '女', 36, '南京', 0),
('C018', '尤磊', '男', 42, '济南', 18),
('C019', '许晴', '女', 50, '天津', 0),
('C020', '何勇', '男', 26, '广州', 3);
'''

# ----------------------------------------------------------------------------------------------------------------------

# 2. 保险产品定义表 (products)
products = '''
CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(32) PRIMARY KEY,       -- 产品唯一ID
    product_name VARCHAR(100) NOT NULL,       -- 产品名称
    aliases VARCHAR(100),                     -- 产品简称
    insurance_type VARCHAR(20) CHECK(insurance_type IN ('医疗险', '重疾险', '车险')), -- 险种分类
    base_premium REAL NOT NULL,               -- 基础保费/年
    waiting_period_days INTEGER DEFAULT 90    -- 等待期天数 (健康险特有，免责期)
);
'''

# 插入 customers 表数据 (20条：包含不同城市、年龄和驾龄)
products_data = '''INSERT INTO products (product_id, product_name, aliases, insurance_type, base_premium, waiting_period_days) VALUES
('P001', '长生爱康保重大疾病保险', '爱康保,长爱保,长生爱康保', '重疾险', 500.0, 90),
('P002', '华泰人寿百万康宁终身重大疾病保险', '百万康宁,华泰康宁', '重疾险', 300.0, 30);
'''

# ----------------------------------------------------------------------------------------------------------------------

# 3. 客户保单表 (policies)
policies = '''CREATE TABLE IF NOT EXISTS policies (
    policy_no VARCHAR(32) PRIMARY KEY,        -- 保单号 (系统唯一单号)
    customer_id VARCHAR(32) NOT NULL,         -- 客户ID (关联customers)
    product_id VARCHAR(32) NOT NULL,          -- 产品ID (关联products)
    premium REAL NOT NULL,                    -- 实际实付保费 (可能根据年龄、驾龄有折扣或加费)
    effective_date DATE NOT NULL,             -- 保单生效日期
    expiry_date DATE NOT NULL,                -- 保单失效日期
    policy_status VARCHAR(20) DEFAULT '有效' CHECK(policy_status IN ('有效', '已满期', '已退保', '失效')), -- 保单状态
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
'''

# 插入 policies 表数据 (14条：包含不同状态、保费折扣/加费、生效失效日期)
policies_data = '''
INSERT INTO policies (policy_no, customer_id, product_id, premium, effective_date, expiry_date, policy_status) VALUES
('POL001', 'C001', 'P002', 2000.0, '2026-01-01', '2027-01-01', '有效'),
('POL002', 'C002', 'P001', 250.0, '2026-03-15', '2027-03-15', '有效'),
('POL003', 'C003', 'P002', 2200.0, '2025-06-01', '2026-06-01', '已满期'),
('POL004', 'C004', 'P001', 400.0, '2026-05-10', '2027-05-10', '有效'),
('POL005', 'C005', 'P002', 1800.0, '2026-08-20', '2027-08-20', '有效'),
('POL006', 'C006', 'P001', 300.0, '2026-02-01', '2027-02-01', '已退保'),
('POL007', 'C007', 'P001', 350.0, '2024-11-11', '2025-11-11', '已满期'),
('POL008', 'C008', 'P002', 2000.0, '2026-07-01', '2027-07-01', '有效'),
('POL009', 'C010', 'P001', 500.0, '2026-01-05', '2027-01-05', '有效'),
('POL010', 'C011', 'P001', 280.0, '2024-09-09', '2025-09-09', '已满期'),
('POL011', 'C012', 'P002', 2100.0, '2026-04-15', '2027-04-15', '有效'),
('POL012', 'C014', 'P002', 1900.0, '2026-06-18', '2027-06-18', '有效'),
('POL013', 'C016', 'P001', 380.0, '2026-09-01', '2027-09-01', '有效'),
('POL014', 'C018', 'P002', 2050.0, '2024-12-25', '2025-12-25', '已满期');

'''

# ----------------------------------------------------------------------------------------------------------------------

# 4. 理赔申请流水表 (claims)
claims = '''CREATE TABLE IF NOT EXISTS claims (
    claim_id VARCHAR(32) PRIMARY KEY,         -- 理赔案件号
    policy_no VARCHAR(32) NOT NULL,           -- 保单号 (关联policies)
    customer_id VARCHAR(32) NOT NULL,         -- 申请人ID (关联customers)
    claim_type VARCHAR(20) CHECK(claim_type IN ('疾病医疗', '重大疾病', '人身意外')), -- 理赔类型
    report_date DATE NOT NULL,                -- 报案日期
    claim_amount REAL NOT NULL,               -- 申请理赔金额
    paid_amount REAL DEFAULT 0.0,             -- 实际赔付金额 (拒赔时为0)
    claim_status VARCHAR(20) DEFAULT '审核中' CHECK(claim_status IN ('审核中', '结案赔付', '拒赔')), -- 理赔状态 (HITL人工介入核心)
    reject_reason TEXT,                       -- 拒赔原因 (如: '属于既往症免责范围', '等待期内确诊')
    FOREIGN KEY (policy_no) REFERENCES policies(policy_no),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
'''

# 插入 claims 表数据 (5条：包含结案赔付、等待期内拒赔、审核中状态)
claims_data = '''
INSERT INTO claims (claim_id, policy_no, customer_id, claim_type, report_date, claim_amount, paid_amount, claim_status, reject_reason) VALUES
('CL001', 'POL001', 'C001', '疾病医疗', '2025-05-12', 5000.0, 4500.0, '结案赔付', NULL),
('CL002', 'POL002', 'C002', '重大疾病', '2025-06-20', 15000.0, 12000.0, '结案赔付', NULL),
('CL003', 'POL004', 'C004', '重大疾病', '2025-06-15', 500000.0, 0.0, '拒赔', '属于等待期内确诊免责范围'),
('CL004', 'POL008', 'C008', '疾病医疗', '2025-11-02', 8000.0, 0.0, '审核中', NULL),
('CL005', 'POL011', 'C012', '人身意外', '2025-11-15', 3000.0, 2500.0, '结案赔付', NULL);
'''


def create_table(cursor):
    cursor.execute(customers)
    cursor.execute(products)
    cursor.execute(policies)
    cursor.execute(claims)


def insert_data(cursor):
    cursor.execute(customers_data)
    cursor.execute(products_data)
    cursor.execute(policies_data)
    cursor.execute(claims_data)


if __name__ == '__main__':
    os.makedirs('../db/sqlite/', exist_ok=True)
    conn = sqlite3.connect("../db/sqlite/insurance.db", check_same_thread=False)
    cursor = conn.cursor()

    print("数据库已创建完毕！")
    create_table(cursor)
    insert_data(cursor)
    conn.commit()
    print("数据已插入完毕！")