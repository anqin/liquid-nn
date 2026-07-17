#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C) 2026 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : spider_discount.py
#   Last Modified : 2026-07-17 16:18
#   Describe      : 
#
# ====================================================

import time
import threading
import argparse
from datetime import datetime
from bs4 import BeautifulSoup
import pymysql
from playwright.sync_api import sync_playwright

# ====================== 配置区（自行修改）======================
# MySQL配置（密码建议纯英文数字，避免编码报错）
MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "abc123456",
    "database": "discount_db",
    "charset": "utf8mb4"
}
# 采集间隔（秒），30分钟=1800秒
SPIDER_INTERVAL = 1800

# 各平台优惠页面链接（自行替换优惠专区）
PLATFORM_URLS = {
    "douyin": "https://mall.douyin.com/",
    "xiaohongshu": "https://www.xiaohongshu.com/mall",
    "ctrip": "https://www.ctrip.com/",
    "meituan": "https://www.meituan.com/",
    "dazhong": "https://www.dianping.com/"
}
# 平台中文名映射
PLAT_NAME_MAP = {
    "douyin": "抖音商城",
    "xiaohongshu": "小红书商城",
    "ctrip": "携程旅行",
    "meituan": "美团",
    "dazhong": "大众点评"
}
# ==============================================================

# -------------------------- 数据库工具 --------------------------
def get_mysql_conn():
    """统一获取数据库连接，兼容中文密码"""
    cfg = MYSQL_CONFIG.copy()
    cfg["password"] = cfg["password"].encode("utf-8")
    conn = pymysql.connect(**cfg)
    return conn

def init_mysql():
    """初始化数据库、数据表"""
    conn = get_mysql_conn()
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']} DEFAULT CHARACTER SET utf8mb4;")
    conn.select_db(MYSQL_CONFIG["database"])
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS discount_info (
        id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
        platform VARCHAR(32) NOT NULL COMMENT '平台标识',
        title VARCHAR(512) NOT NULL COMMENT '商品/团购标题',
        original_price DECIMAL(10,2) DEFAULT 0 COMMENT '原价',
        sale_price DECIMAL(10,2) DEFAULT 0 COMMENT '优惠现价',
        discount_text VARCHAR(256) COMMENT '优惠文案（满减/优惠券）',
        goods_url TEXT COMMENT '商品详情链接',
        collect_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '采集时间',
        UNIQUE KEY unique_url (goods_url(255))
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='各平台优惠数据表';
    """
    cursor.execute(create_table_sql)
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ MySQL数据库&数据表初始化完成")

def save_to_mysql(data_list):
    """批量写入数据库，重复链接自动忽略"""
    if not data_list:
        return
    conn = get_mysql_conn()
    cursor = conn.cursor()
    insert_sql = """
    INSERT IGNORE INTO discount_info
    (platform, title, original_price, sale_price, discount_text, goods_url)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    batch_params = []
    for item in data_list:
        batch_params.append((
            item["platform"],
            item["title"],
            item["original_price"],
            item["sale_price"],
            item["discount_text"],
            item["goods_url"]
        ))
    cursor.executemany(insert_sql, batch_params)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ 本次成功入库 {len(data_list)} 条优惠数据")

def query_discount(platform=None, keyword=None, limit=50):
    """数据库查询接口"""
    conn = get_mysql_conn()
    cursor = conn.cursor()
    base_sql = "SELECT platform,title,original_price,sale_price,discount_text,collect_time,goods_url FROM discount_info WHERE 1=1 "
    params = []
    if platform:
        base_sql += " AND platform=%s "
        params.append(platform)
    if keyword:
        base_sql += " AND title LIKE %s "
        params.append(f"%{keyword}%")
    base_sql += " ORDER BY collect_time DESC LIMIT %s"
    params.append(limit)
    cursor.execute(base_sql, params)
    res = cursor.fetchall()
    cursor.close()
    conn.close()
    return res

def print_query_result(data):
    """格式化打印查询结果"""
    if not data:
        print("\n🔍 未查询到匹配优惠数据")
        return
    print(f"\n==================== 查询结果（共{len(data)}条）====================")
    for idx, row in enumerate(data, 1):
        plat, title, ori, sale, discount, ctime, url = row
        cname = PLAT_NAME_MAP.get(plat, plat)
        print(f"\n【{idx}】平台：{cname}")
        print(f"标题：{title}")
        print(f"原价：¥{ori:.2f} | 现价：¥{sale:.2f} | 优惠：{discount}")
        print(f"采集时间：{ctime}")
        print(f"链接：{url[:100]}..." if len(url) > 100 else f"链接：{url}")
    print("==================================================================\n")

# -------------------------- Playwright 页面采集封装（替代Selenium） --------------------------
def fetch_page_html(page, url):
    """访问页面，等待动态加载，返回html"""
    # 手机UA模拟
    mobile_ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    page.set_extra_http_headers({"User-Agent": mobile_ua})
    # 访问页面
    page.goto(url, timeout=30000, wait_until="networkidle")
    time.sleep(3)
    return page.content()

def parse_platform(page, platform_name, url):
    print(f"\n===== 开始采集【{PLAT_NAME_MAP.get(platform_name, platform_name)}】优惠 =====")
    try:
        html = fetch_page_html(page, url)
    except Exception as e:
        print(f"{platform_name} 页面加载失败：{str(e)}")
        return []

    soup = BeautifulSoup(html, "lxml")
    item_cards = soup.select("div[class*='item'], div[class*='goods'], div[class*='card']")
    discount_list = []
    for card in item_cards[:30]:
        try:
            # 标题&链接
            title_tag = card.select_one("a[href], h3, p[class*='title']")
            title = title_tag.get_text(strip=True) if title_tag else "无标题"
            goods_url = ""
            if title_tag and title_tag.has_attr("href"):
                goods_url = title_tag["href"]
                if goods_url.startswith("/"):
                    goods_url = url.rstrip("/") + goods_url
            # 优惠现价
            sale_tag = card.select_one("span[class*='price'], span[class*='sale'], em")
            sale_price = 0.0
            if sale_tag:
                price_text = sale_tag.get_text(strip=True).replace("¥", "").replace("￥", "").replace(",", "")
                if price_text.replace(".", "").isdigit():
                    sale_price = float(price_text)
            # 原价
            origin_tag = card.select_one("span[class*='origin'], span[class*='old'], del")
            original_price = sale_price
            if origin_tag:
                ori_text = origin_tag.get_text(strip=True).replace("¥", "").replace("￥", "").replace(",", "")
                if ori_text.replace(".", "").isdigit():
                    original_price = float(ori_text)
            # 优惠标签
            discount_tag = card.select_one("span[class*='coupon'], span[class*='discount'], div[class*='tag']")
            discount_text = discount_tag.get_text(strip=True) if discount_tag else "无优惠券"

            if sale_price <= 0 or title == "无标题":
                continue
            discount_list.append({
                "platform": platform_name,
                "title": title,
                "original_price": original_price,
                "sale_price": sale_price,
                "discount_text": discount_text,
                "goods_url": goods_url
            })
        except Exception:
            continue
    print(f"【{PLAT_NAME_MAP.get(platform_name, platform_name)}】本次抓取 {len(discount_list)} 条优惠")
    return discount_list

def run_single_spider():
    """一轮完整采集，使用playwright"""
    all_data = []
    try:
        with sync_playwright() as p:
            # 启动无头chromium，内置反爬，无需额外配置
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu"
                ]
            )
            context = browser.new_context(
                # 自动隐藏webdriver检测，原生反爬
                bypass_csp=True
            )
            page = context.new_page()
            # 遍历所有平台
            for plat, url in PLATFORM_URLS.items():
                items = parse_platform(page, plat, url)
                all_data.extend(items)
            browser.close()
    except Exception as err:
        print(f"采集任务全局异常：{str(err)}")
    save_to_mysql(all_data)
    print(f"\n⏰ 等待 {SPIDER_INTERVAL/60:.1f} 分钟后执行下一轮采集...")
    # 定时循环
    threading.Timer(SPIDER_INTERVAL, run_single_spider).start()

# -------------------------- 命令行入口主逻辑 --------------------------
def main():
    parser = argparse.ArgumentParser(description="多平台优惠定时采集工具 | 爬虫采集 / 数据库查询")
    subparsers = parser.add_subparsers(dest="cmd", required=True, help="运行模式")

    # 子命令1：定时爬虫
    parser_spider = subparsers.add_parser("spider", help="启动定时爬虫，持续采集优惠入库")

    # 子命令2：查询数据
    parser_query = subparsers.add_parser("query", help="查询本地数据库优惠数据")
    parser_query.add_argument("-p", "--platform", type=str, choices=list(PLATFORM_URLS.keys()), help="平台：douyin/xiaohongshu/ctrip/meituan/dazhong")
    parser_query.add_argument("-k", "--keyword", type=str, help="标题模糊搜索关键词")
    parser_query.add_argument("-l", "--limit", type=int, default=50, help="最大返回条数，默认50")

    args = parser.parse_args()

    if args.cmd == "spider":
        print("=== 多平台优惠定时爬虫启动 ===")
        init_mysql()
        run_single_spider()
        # 主线程保活
        while True:
            time.sleep(60)

    elif args.cmd == "query":
        print("=== 本地优惠数据库查询 ===")
        res = query_discount(platform=args.platform, keyword=args.keyword, limit=args.limit)
        print_query_result(res)

if __name__ == "__main__":
    main()
