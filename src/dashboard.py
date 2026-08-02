import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.logger import logger
from src.config import (
    PROCESSED_DATA_DIR,
    DASHBOARD_DIR
)


class Dashboard:

    def __init__(self):

        self.file = (
            PROCESSED_DATA_DIR /
            "conformal_predictions.csv"
        )

        self.output = (
            DASHBOARD_DIR /
            "dashboard.html"
        )

    ############################################################

    def load_dataset(self):

        logger.info(
            "Loading Dashboard Dataset..."
        )

        df = pd.read_csv(
            self.file
        )

        logger.success(
            f"Dataset : {df.shape}"
        )

        return df

    ############################################################

    def create_dashboard(
        self,
        df
    ):

        logger.info(
            "Building Dashboard..."
        )

        avg_return = round(
            df["Return"].mean(),
            5
        )

        avg_volatility = round(
            df["Volatility"].mean(),
            5
        )

        avg_confidence = round(
            df["Confidence"].mean(),
            3
        )

        avg_risk = round(
            df["Risk_Score"].mean(),
            3
        )

        total_days = len(df)

        regimes = df["Final_Regime"].nunique()

        fig = make_subplots(

            rows=4,

            cols=2,

            specs=[

                [{"type":"xy"},{"type":"xy"}],

                [{"type":"xy"},{"type":"xy"}],

                [{"type":"xy"},{"type":"domain"}],

                [{"type":"table"},{"type":"table"}]

            ],

            subplot_titles=(

                "Market Close Price",

                "Regime Distribution",

                "Volatility",

                "Prediction Confidence",

                "Risk Score",

                "Regime Percentage",

                "Recent Predictions",

                "Model Summary"

            )

        )

        ########################################################
        # KPI Header
        ########################################################

        fig.add_annotation(

            text=f"""

<b>Bayesian Regime Detection Engine</b><br><br>

Trading Days : <b>{total_days}</b>

&nbsp;&nbsp;&nbsp;&nbsp;

Hidden Regimes : <b>{regimes}</b>

&nbsp;&nbsp;&nbsp;&nbsp;

Average Return : <b>{avg_return}</b>

&nbsp;&nbsp;&nbsp;&nbsp;

Average Volatility : <b>{avg_volatility}</b>

&nbsp;&nbsp;&nbsp;&nbsp;

Average Confidence : <b>{avg_confidence}</b>

&nbsp;&nbsp;&nbsp;&nbsp;

Average Risk : <b>{avg_risk}</b>

""",

            x=0.5,

            y=1.15,

            showarrow=False,

            xref="paper",

            yref="paper",

            font=dict(

                size=16

            )

        )

        ########################################################
        # Close Price
        ########################################################

        fig.add_trace(

            go.Scatter(

                x=df["Date"],

                y=df["Close"],

                mode="lines",

                name="Close",

                line=dict(width=2)

            ),

            row=1,

            col=1

        )
            ########################################################
        # Regime Distribution
        ########################################################

        regime_count = (

            df["Final_Regime"]

            .value_counts()

            .sort_index()

        )

        fig.add_trace(

            go.Bar(

                x=regime_count.index.astype(str),

                y=regime_count.values,

                name="Regimes"

            ),

            row=1,

            col=2

        )

        ########################################################
        # Volatility
        ########################################################

        fig.add_trace(

            go.Scatter(

                x=df["Date"],

                y=df["Volatility"],

                mode="lines",

                name="Volatility",

                line=dict(width=2)

            ),

            row=2,

            col=1

        )

        ########################################################
        # Confidence
        ########################################################

        fig.add_trace(

            go.Scatter(

                x=df["Date"],

                y=df["Confidence"],

                mode="lines",

                name="Confidence",

                line=dict(width=2)

            ),

            row=2,

            col=2

        )

        ########################################################
        # Risk Score
        ########################################################

        fig.add_trace(

            go.Scatter(

                x=df["Date"],

                y=df["Risk_Score"],

                mode="lines",

                name="Risk Score",

                line=dict(width=2)

            ),

            row=3,

            col=1

        )

        ########################################################
        # Regime Percentage
        ########################################################

        fig.add_trace(

            go.Pie(

                labels=regime_count.index.astype(str),

                values=regime_count.values,

                hole=0.45,

                showlegend=False

            ),

            row=3,

            col=2

        )
            ########################################################
        # Recent Predictions Table
        ########################################################

        recent = df.tail(20)

        fig.add_trace(

            go.Table(

                header=dict(

                    values=[
                        "Date",
                        "Close",
                        "Regime",
                        "Confidence"
                    ]

                ),

                cells=dict(

                    values=[

                        recent["Date"],

                        recent["Close"].round(2),

                        recent["Final_Regime"],

                        recent["Confidence"].round(3)

                    ]

                )

            ),

            row=4,

            col=1

        )

        ########################################################
        # Model Summary
        ########################################################

        summary = [

            ["Trading Days", len(df)],

            ["Hidden Regimes", df["Final_Regime"].nunique()],

            ["Average Return", round(df["Return"].mean(),5)],

            ["Average Volatility", round(df["Volatility"].mean(),5)],

            ["Average Confidence", round(df["Confidence"].mean(),3)],

            ["Average Risk", round(df["Risk_Score"].mean(),3)]

        ]

        fig.add_trace(

            go.Table(

                header=dict(

                    values=["Metric","Value"]

                ),

                cells=dict(

                    values=[

                        [x[0] for x in summary],

                        [x[1] for x in summary]

                    ]

                )

            ),

            row=4,

            col=2

        )

        ########################################################
        # Dashboard Layout
        ########################################################

        fig.update_layout(

            title={

                "text":"Bayesian Regime Detection Dashboard",

                "x":0.5

            },

            template="plotly_white",

            height=1500,

            width=1500,

            hovermode="x unified"

        )

        ########################################################
        # Footer
        ########################################################

        fig.add_annotation(

            text="Developed by Md Mokarram Ali",

            x=0.5,

            y=-0.08,

            xref="paper",

            yref="paper",

            showarrow=False,

            font=dict(size=14)

        )

        ########################################################
        # Save Dashboard
        ########################################################

        fig.write_html(

            self.output

        )

        logger.success(

            f"Dashboard Saved : {self.output}"

        )

    ############################################################
    # Run
    ############################################################

    def run(self):

        logger.info("="*70)

        logger.info(
            "Starting Dashboard..."
        )

        logger.info("="*70)

        df = self.load_dataset()

        self.create_dashboard(df)

        logger.success("="*70)

        logger.success(
            "Dashboard Created Successfully."
        )

        logger.success("="*70)
        