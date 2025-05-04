"""Data pipeline for collecting and organizing EUI simulation results."""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from supabase import create_client, Client


class EUIDataPipeline:
    """
    Manages the collection (via Azure) and preparation (from Supabase)
    of EUI simulation data.
    """

    def __init__(self, config: dict):
        """
        Initialize the EUI data pipeline.

        Args:
            config (dict): Configuration dictionary containing paths, Azure settings,
                           and Supabase connection details (e.g., under a 'supabase' key).
                           Expected Supabase keys: 'url', 'key', 'table'.
        """
        self.config = config

        # --- 本地结果目录 ---
        self.results_dir = Path(
            config['paths']['results_dir']) / 'EUI_Data'  # 确保路径存在
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # --- Supabase 配置 ---
        self.supabase_url: Optional[str] = config.get(
            'supabase', {}).get('url')
        self.supabase_key: Optional[str] = config.get(
            'supabase', {}).get('key')
        self.supabase_table: Optional[str] = config.get(
            'supabase', {}).get('table')
        self.supabase: Optional[Client] = None  # Supabase 客户端

        if not all([self.supabase_url, self.supabase_key, self.supabase_table]):
            logging.warning("Supabase URL, Key, or Table not fully configured. "
                            "Data fetching from Supabase will not be possible.")
        else:
            self._connect_supabase()  # 尝试在初始化时连接

    def _connect_supabase(self):
        """尝试连接到 Supabase 数据库。"""
        if not self.supabase_url or not self.supabase_key:
            logging.error("Supabase URL or Key is missing, cannot connect.")
            return False
        if self.supabase:
            logging.debug("Already connected to Supabase.")
            return True
        try:
            logging.info(f"Connecting to Supabase at {self.supabase_url}...")
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            # 可以添加一个简单的测试查询来验证连接
            self.supabase.table(self.supabase_table).select('id', count='exact').limit(0).execute()
            logging.info("Successfully connected to Supabase.")
            return True
        except Exception as e:
            logging.error(f"Error connecting to Supabase: {e}")
            self.supabase = None  # 确保连接失败时客户端为 None
            return False

    def prepare_training_data(self, cities: List[str], ssps: List[int], btypes: List[str]) -> pd.DataFrame:
        """
        从 Supabase 数据库准备（获取）用于神经网络模型的训练数据。

        Args:
            cities (List[str]): 需要包含的城市列表。
            ssps (List[int]): 需要包含的 SSP 场景列表 (应与 Supabase 表中的 ssp 列匹配)。
                               假设 Supabase 中的列名为 'ssp_code' 或 'ssp'。
            btypes (List[str]): 需要包含的建筑类型列表。

        Returns:
            pd.DataFrame: 从 Supabase 获取并组合的训练数据。如果出错或未找到数据，则返回空 DataFrame。
        """
        if not self.supabase:
            logging.error(
                "Supabase client is not connected. Cannot prepare training data.")
            # 尝试重新连接
            if not self._connect_supabase():
                return pd.DataFrame()  # 如果连接失败，返回空 DataFrame

        if not self.supabase_table:
            logging.error("Supabase table name is not configured.")
            return pd.DataFrame()

        logging.info(
            f"Preparing training data from Supabase table '{self.supabase_table}'...")
        logging.info(
            f"Filters: cities={cities}, ssps={ssps}, btypes={btypes}")

        try:
            all_data = []
            page_size = 1000  # 每次获取的行数，通常设为默认值或 Supabase 的 max_rows
            current_page = 0
            fetched_count_last_page = page_size # 用于控制循环

            logging.info(f"Starting paginated fetch from Supabase (page size: {page_size})...")

            while fetched_count_last_page == page_size: # 只有当上一页获取满了才可能需要获取下一页
                start_row = current_page * page_size
                end_row = start_row + page_size - 1
                logging.debug(f"Fetching page {current_page + 1} (rows {start_row}-{end_row})...")

                # 重新构建基础查询或确保过滤器在分页前应用
                query_builder = self.supabase.table(self.supabase_table).select("*")

                # 应用你的过滤器
                if cities:
                    query_cities = [city.upper() for city in cities]
                    query_builder = query_builder.in_('city', query_cities)
                if ssps:
                    query_ssps = [ssp for ssp in ssps]
                    query_builder = query_builder.in_('ssp_code', query_ssps) # 确认列名是 ssp_code
                if btypes:
                    query_btypes = [btype.upper() for btype in btypes]
                    query_builder = query_builder.in_('btype', query_btypes)

                # 应用分页
                response = query_builder.range(start_row, end_row).execute()

                if hasattr(response, 'data') and response.data:
                    page_data = response.data
                    fetched_count_last_page = len(page_data)
                    all_data.extend(page_data)
                    logging.debug(f"Fetched {fetched_count_last_page} records for this page.")
                    # 如果获取到的记录数小于页面大小，说明这是最后一页了
                    if fetched_count_last_page < page_size:
                        break
                    current_page += 1 # 准备获取下一页
                elif hasattr(response, 'error') and response.error:
                    logging.error(f"Error fetching data from Supabase on page {current_page + 1}: {response.error}")
                    fetched_count_last_page = 0 # 出错时终止循环
                else:
                    # 没有数据或意外响应
                    logging.warning(f"No data or unexpected response on page {current_page + 1}. Stopping fetch.")
                    fetched_count_last_page = 0 # 终止循环


            combined_df = pd.DataFrame(all_data).drop(columns=['created_at'])
            logging.info(
                f"Successfully fetched {len(combined_df)} records from Supabase.")

            output_file = self.results_dir / "training_data_from_database.csv"
            combined_df.to_csv(output_file, index=False)
            logging.info(f"Saved fetched training data to {output_file}")

            return combined_df

        except Exception as e:
            logging.error(
                f"An unexpected error occurred while querying Supabase: {e}")
            # 可以考虑记录更详细的堆栈跟踪信息
            # import traceback
            # logging.error(traceback.format_exc())
            return pd.DataFrame()  # 发生异常，返回空 DataFrame

    def load_data(self) -> pd.DataFrame:
        """
        从本地文件加载已准备好的训练数据。

        Args:
            source (str): 指定数据来源的文件名后缀，例如 "supabase" 或 "azure"。
                          默认为 "supabase"，对应 `prepare_training_data` 保存的文件。

        Returns:
            pd.DataFrame: 加载的训练数据。

        Raises:
            FileNotFoundError: 如果对应的本地训练数据文件不存在。
        """
        data_filename = "training_data_from_database.csv"

        data_file = self.results_dir / data_filename

        if not data_file.exists():
            raise FileNotFoundError(f"Data file '{data_file}' not found. "
                                    f"Run prepare_training_data (for Supabase) or ensure '{data_filename}' data exists.")

        logging.info(f"Loading Data from {data_file}")
        return pd.read_csv(data_file)
