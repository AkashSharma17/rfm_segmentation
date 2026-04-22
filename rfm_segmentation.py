import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class CustomerSegmentationRFM:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.rfm = None

    # =========================
    # LOAD
    # =========================
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print("✅ Data Loaded")
        except Exception as e:
            raise Exception(f"❌ Error loading file: {e}")

    # =========================
    # VALIDATE
    # =========================
    def validate(self):
        required = ["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]

        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"❌ Missing column: {col}")

        print("✅ Validation Passed")

    # =========================
    # CLEAN
    # =========================
    def clean_data(self):
        self.df.drop_duplicates(inplace=True)

        self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"], errors="coerce")
        self.df["Quantity"] = pd.to_numeric(self.df["Quantity"], errors="coerce")
        self.df["UnitPrice"] = pd.to_numeric(self.df["UnitPrice"], errors="coerce")

        self.df.dropna(inplace=True)
        self.df = self.df[(self.df["Quantity"] > 0) & (self.df["UnitPrice"] > 0)]

        self.df["TotalPrice"] = self.df["Quantity"] * self.df["UnitPrice"]

        print("✅ Data Cleaned")

    # =========================
    # RFM
    # =========================
    def calculate_rfm(self):
        snapshot = self.df["InvoiceDate"].max() + pd.Timedelta(days=1)

        self.rfm = self.df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (snapshot - x.max()).days,
            "InvoiceNo": "count",
            "TotalPrice": "sum"
        })

        self.rfm.columns = ["Recency", "Frequency", "Monetary"]

        print("✅ RFM Calculated")

    # =========================
    # SCORING (SAFE)
    # =========================
    def score_rfm(self):
        try:
            self.rfm["R"] = pd.qcut(self.rfm["Recency"], 4, labels=[4,3,2,1])
            self.rfm["F"] = pd.qcut(self.rfm["Frequency"].rank(method="first"), 4, labels=[1,2,3,4])
            self.rfm["M"] = pd.qcut(self.rfm["Monetary"], 4, labels=[1,2,3,4])
        except Exception:
            self.rfm["R"] = pd.cut(self.rfm["Recency"], 4, labels=[4,3,2,1])
            self.rfm["F"] = pd.cut(self.rfm["Frequency"], 4, labels=[1,2,3,4])
            self.rfm["M"] = pd.cut(self.rfm["Monetary"], 4, labels=[1,2,3,4])

        self.rfm["RFM_Score"] = (
            self.rfm["R"].astype(str) +
            self.rfm["F"].astype(str) +
            self.rfm["M"].astype(str)
        )

        print("✅ RFM Scored")

    # =========================
    # SEGMENT
    # =========================
    def segment(self):
        def label(row):
            if row["RFM_Score"] == "444":
                return "VIP"
            elif int(row["F"]) == 4:
                return "Loyal"
            elif int(row["R"]) == 4:
                return "Recent"
            elif int(row["R"]) == 1:
                return "At Risk"
            else:
                return "Regular"

        self.rfm["Segment"] = self.rfm.apply(label, axis=1)

        print("✅ Segmentation Done")

    # =========================
    # PARETO
    # =========================
    def pareto(self):
        self.rfm = self.rfm.sort_values(by="Monetary", ascending=False)

        self.rfm["Cumulative_Percentage"] = (
            self.rfm["Monetary"].cumsum() / self.rfm["Monetary"].sum()
        )

        self.rfm["Top_Customers"] = self.rfm["Cumulative_Percentage"] <= 0.8

        print("✅ Pareto Analysis Done")

    # =========================
    # INSIGHTS
    # =========================
    def insights(self):
        print("\n📊 BUSINESS INSIGHTS\n")

        print("Revenue by Segment:\n")
        print(self.rfm.groupby("Segment")["Monetary"].sum())

        top = self.rfm[self.rfm["Top_Customers"]]
        print(f"\nTop Customers Count: {len(top)}")
        print(f"Revenue from Top Customers: {top['Monetary'].sum():.2f}")

    # =========================
    # VISUALIZATION (PRO)
    # =========================
    def visualize(self):
        counts = self.rfm["Segment"].value_counts()

        plt.figure()
        counts.plot(kind="bar")

        plt.title("Customer Segmentation Distribution")
        plt.xlabel("Customer Segment")
        plt.ylabel("Number of Customers")

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        plt.show()

    # =========================
    # EXPORT
    # =========================
    def export(self):
        os.makedirs("output", exist_ok=True)
        self.rfm.to_csv("output/rfm_output.csv", index=True)
        print("✅ File Saved in output/rfm_output.csv")

    # =========================
    # RUN
    # =========================
    def run(self):
        self.load_data()
        self.validate()
        self.clean_data()
        self.calculate_rfm()
        self.score_rfm()
        self.segment()
        self.pareto()
        self.insights()
        self.visualize()
        self.export()


if __name__ == "__main__":
    obj = CustomerSegmentationRFM("ecommerce_data.csv")
    obj.run()