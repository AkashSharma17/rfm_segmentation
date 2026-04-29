import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# MAIN CLASS
# =========================================================

class CustomerSegmentationRFM:

    # =====================================================
    # INIT
    # =====================================================

    def __init__(self, file_path: str):

        logger.info("Initializing RFM Analysis Project")

        self.file_path = file_path
        self.df = None
        self.rfm = None

    # =====================================================
    # LOAD DATA
    # =====================================================

    def load_data(self):

        logger.info("Loading Dataset")

        try:

            self.df = pd.read_csv(self.file_path)

            logger.info(
                f"Dataset Loaded Successfully | Shape: {self.df.shape}"
            )

        except Exception as e:

            logger.error(f"Error Loading Dataset: {e}")

            raise

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate_data(self):

        logger.info("Validating Dataset")

        required_columns = [

            "CustomerID",
            "InvoiceNo",
            "InvoiceDate",
            "Quantity",
            "UnitPrice"
        ]

        missing_columns = [

            col for col in required_columns
            if col not in self.df.columns
        ]

        if missing_columns:

            logger.error(
                f"Missing Columns Found: {missing_columns}"
            )

            raise ValueError(
                f"Dataset Missing Columns: {missing_columns}"
            )

        logger.info("Dataset Validation Successful")

    # =====================================================
    # DATA CLEANING
    # =====================================================

    def clean_data(self):

        logger.info("Starting Data Cleaning")

        before_rows = len(self.df)

        # Remove Duplicates
        duplicates = self.df.duplicated().sum()

        self.df.drop_duplicates(inplace=True)

        logger.info(
            f"Duplicate Rows Removed: {duplicates}"
        )

        # Convert Data Types
        self.df["InvoiceDate"] = pd.to_datetime(
            self.df["InvoiceDate"],
            errors="coerce"
        )

        self.df["Quantity"] = pd.to_numeric(
            self.df["Quantity"],
            errors="coerce"
        )

        self.df["UnitPrice"] = pd.to_numeric(
            self.df["UnitPrice"],
            errors="coerce"
        )

        logger.info("Data Type Conversion Completed")

        # Missing Values
        missing_values = self.df.isna().sum().sum()

        logger.info(
            f"Total Missing Values Found: {missing_values}"
        )

        self.df.dropna(inplace=True)

        # Remove Invalid Values
        self.df = self.df[
            (self.df["Quantity"] > 0) &
            (self.df["UnitPrice"] > 0)
        ]

        logger.info(
            "Invalid Quantity and Price Rows Removed"
        )

        # Feature Engineering
        self.df["TotalPrice"] = (
            self.df["Quantity"] *
            self.df["UnitPrice"]
        )

        after_rows = len(self.df)

        logger.info(
            f"Cleaning Completed | Before: {before_rows} | After: {after_rows}"
        )

    # =====================================================
    # BASIC EDA
    # =====================================================

    def basic_eda(self):

        logger.info("Running Basic EDA")

        logger.info(f"Dataset Shape: {self.df.shape}")

        logger.info(
            f"Duplicate Rows: {self.df.duplicated().sum()}"
        )

        logger.info(
            f"Missing Values:\n{self.df.isna().sum()}"
        )

        logger.info(
            f"Data Types:\n{self.df.dtypes}"
        )

        logger.info(
            f"Total Revenue: {self.df['TotalPrice'].sum():.2f}"
        )

    # =====================================================
    # CALCULATE RFM
    # =====================================================

    def calculate_rfm(self):

        logger.info("Calculating RFM Metrics")

        snapshot_date = (
            self.df["InvoiceDate"].max() +
            pd.Timedelta(days=1)
        )

        self.rfm = self.df.groupby("CustomerID").agg({

            "InvoiceDate": lambda x:
            (snapshot_date - x.max()).days,

            "InvoiceNo": "count",

            "TotalPrice": "sum"

        })

        self.rfm.columns = [

            "Recency",
            "Frequency",
            "Monetary"
        ]

        logger.info("RFM Metrics Calculated")

    # =====================================================
    # RFM SCORING
    # =====================================================

    def score_rfm(self):

        logger.info("Generating RFM Scores")

        try:

            self.rfm["R"] = pd.qcut(
                self.rfm["Recency"],
                4,
                labels=[4, 3, 2, 1]
            )

            self.rfm["F"] = pd.qcut(
                self.rfm["Frequency"].rank(method="first"),
                4,
                labels=[1, 2, 3, 4]
            )

            self.rfm["M"] = pd.qcut(
                self.rfm["Monetary"],
                4,
                labels=[1, 2, 3, 4]
            )

        except Exception:

            logger.warning(
                "qcut failed. Switching to cut()"
            )

            self.rfm["R"] = pd.cut(
                self.rfm["Recency"],
                4,
                labels=[4, 3, 2, 1]
            )

            self.rfm["F"] = pd.cut(
                self.rfm["Frequency"],
                4,
                labels=[1, 2, 3, 4]
            )

            self.rfm["M"] = pd.cut(
                self.rfm["Monetary"],
                4,
                labels=[1, 2, 3, 4]
            )

        self.rfm["RFM_Score"] = (

            self.rfm["R"].astype(str) +
            self.rfm["F"].astype(str) +
            self.rfm["M"].astype(str)

        )

        logger.info("RFM Scoring Completed")

    # =====================================================
    # CUSTOMER SEGMENTATION
    # =====================================================

    def segment_customers(self):

        logger.info("Running Customer Segmentation")

        def segment(row):

            if row["RFM_Score"] == "444":
                return "VIP Customers"

            elif int(row["F"]) == 4:
                return "Loyal Customers"

            elif int(row["R"]) == 4:
                return "Recent Customers"

            elif int(row["R"]) == 1:
                return "At Risk Customers"

            else:
                return "Regular Customers"

        self.rfm["Segment"] = self.rfm.apply(
            segment,
            axis=1
        )

        logger.info("Customer Segmentation Completed")

    # =====================================================
    # PARETO ANALYSIS
    # =====================================================

    def pareto_analysis(self):

        logger.info("Running Pareto Analysis")

        self.rfm = self.rfm.sort_values(
            by="Monetary",
            ascending=False
        )

        self.rfm["Cumulative_Revenue"] = (

            self.rfm["Monetary"].cumsum() /
            self.rfm["Monetary"].sum()

        )

        self.rfm["Top_80_Percent"] = (
            self.rfm["Cumulative_Revenue"] <= 0.80
        )

        logger.info("Pareto Analysis Completed")

    # =====================================================
    # BUSINESS INSIGHTS
    # =====================================================

    def business_insights(self):

        logger.info("Generating Business Insights")

        revenue_by_segment = (

            self.rfm.groupby("Segment")["Monetary"]
            .sum()
            .sort_values(ascending=False)

        )

        logger.info(
            f"Revenue by Segment:\n{revenue_by_segment}"
        )

        top_customers = self.rfm[
            self.rfm["Top_80_Percent"]
        ]

        logger.info(
            f"Top Customers Count: {len(top_customers)}"
        )

        logger.info(
            f"Revenue Generated by Top Customers: "
            f"{top_customers['Monetary'].sum():.2f}"
        )

    # =====================================================
    # VISUALIZATION
    # =====================================================

    def visualize_segments(self):

        logger.info(
            "Generating Customer Segmentation Plot"
        )

        segment_counts = (
            self.rfm["Segment"]
            .value_counts()
        )

        plt.figure(figsize=(8, 4))

        segment_counts.plot(kind="bar")

        plt.title(
            "Customer Segmentation Distribution",
            fontweight="bold"
        )

        plt.xlabel("Customer Segment")

        plt.ylabel("Number of Customers")

        plt.xticks(rotation=15)

        plt.tight_layout()

        plt.show()

        logger.info("Visualization Displayed")

    # =====================================================
    # EXPORT DATA
    # =====================================================

    def export_data(self):

        logger.info("Exporting Final Dataset")

        os.makedirs("output", exist_ok=True)

        self.rfm.to_csv(
            "output/rfm_output.csv"
        )

        logger.info(
            "RFM Output Saved Successfully"
        )

    # =====================================================
    # COMPLETE PIPELINE
    # =====================================================

    def run_pipeline(self):

        logger.info("Project Execution Started")

        self.load_data()

        self.validate_data()

        self.clean_data()

        self.basic_eda()

        self.calculate_rfm()

        self.score_rfm()

        self.segment_customers()

        self.pareto_analysis()

        self.business_insights()

        self.visualize_segments()

        self.export_data()

        logger.info(
            "Project Execution Completed Successfully"
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    project = CustomerSegmentationRFM(
        "ecommerce_data.csv"
    )

    project.run_pipeline()