import os
import dotenv
import influxdb_client

# 初始化
dotenv.load_dotenv(".env")
influx_token = os.getenv("INFLUX_TOKEN")
influx_url = os.getenv("INFLUX_URL")
influx_organization = os.getenv("INFLUX_ORG")
# 序列数据存储
class sequence_storage:
    def __init__(self):
        self.client = influxdb_client.InfluxDBClient(url=influx_url, token=influx_token, org=influx_organization)
        # 不同的api
        self.query_api = self.client.query_api()
        self.write_api = self.client.write_api()
    def create_database(self, db_name:str):
        pass
if __name__ == "__main__":
    ss = sequence_storage()