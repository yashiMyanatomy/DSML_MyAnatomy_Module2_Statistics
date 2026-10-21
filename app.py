
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analysis import (
    run_analysis,
    get_confidence_interval,
)


st.set_page_config(
    page_title="E-Commerce A/B Testing",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 E-Commerce A/B Testing Dashboard")
st.write(
    "Compare the old and new landing pages using "
    "customer conversion data."
)


try:
    df, summary, tests = run_analysis()
    ci_low, ci_high = get_confidence_interval(summary)

except Exception as error:
    st.error(f"Unable to run the analysis: {error}")
    st.stop()


with st.expander("Inspect the cleaned dataset"):
    st.write("Dataset shape:", df.shape)
    st.dataframe(df.head(10), use_container_width=True)

    st.write("Missing values:")
    st.dataframe(df.isna().sum())



#Calculate key metrics

control_rate = summary.loc[
    "control", "conversion_rate"
]

treatment_rate = summary.loc[
    "treatment", "conversion_rate"
]

absolute_lift = treatment_rate - control_rate

relative_lift = (
    (absolute_lift / control_rate) * 100
    if control_rate != 0
    else None
)


# 5. Display KPI cards

st.subheader("Experiment Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Control Conversion Rate",
    f"{control_rate:.2%}",
)

col2.metric(
    "Treatment Conversion Rate",
    f"{treatment_rate:.2%}",
    delta=f"{absolute_lift * 100:+.2f} percentage points",
)

col3.metric(
    "Relative Conversion Lift",
    (
        f"{relative_lift:+.2f}%"
        if relative_lift is not None
        else "N/A"
    ),
)


st.subheader("Conversion Summary")

display_summary = summary[
    [
        "visitors",
        "conversions",
        "non_conversions",
        "conversion_pct",
    ]
].rename(
    columns={
        "visitors": "Visitors",
        "conversions": "Conversions",
        "non_conversions": "Non-conversions",
        "conversion_pct": "Conversion Rate (%)",
    }
)

st.dataframe(
    display_summary.round(3),
    use_container_width=True,
)


st.subheader("1. Number of Visitors")

fig, ax = plt.subplots(figsize=(7, 4))

sns.barplot(
    data=summary.reset_index(),
    x="group",
    y="visitors",
    hue="group",
    legend=False,
    ax=ax,
)

ax.set_xlabel("Experiment Group")
ax.set_ylabel("Unique Visitors")
ax.set_title("Visitors in Each Group")

st.pyplot(fig)
plt.close(fig)



st.subheader("2. Conversion Outcomes")

fig, ax = plt.subplots(figsize=(7, 4))

sns.countplot(
    data=df,
    x="group",
    hue="converted",
    order=["control", "treatment"],
    hue_order=[0, 1],
    ax=ax,
)

ax.set_xlabel("Experiment Group")
ax.set_ylabel("Number of Users")
ax.set_title("Converted vs. Not Converted")
ax.legend(
    title="Converted",
    labels=["No", "Yes"],
)

st.pyplot(fig)
plt.close(fig)


st.subheader("3. Conversion Rate Comparison")

fig, ax = plt.subplots(figsize=(7, 4))

sns.barplot(
    data=summary.reset_index(),
    x="group",
    y="conversion_pct",
    hue="group",
    legend=False,
    ax=ax,
)

ax.set_xlabel("Experiment Group")
ax.set_ylabel("Conversion Rate (%)")
ax.set_title("Old Page vs. New Page")

st.pyplot(fig)
plt.close(fig)


#Chart: daily conversion trends

st.subheader("4. Daily Conversion Trends")

daily = (
    df.assign(date=df["timestamp"].dt.date)
    .groupby(["date", "group"])["converted"]
    .mean()
    .reset_index(name="conversion_rate")
)

daily["conversion_pct"] = (
    daily["conversion_rate"] * 100
)

fig, ax = plt.subplots(figsize=(10, 4))

sns.lineplot(
    data=daily,
    x="date",
    y="conversion_pct",
    hue="group",
    marker="o",
    ax=ax,
)

ax.set_xlabel("Date")
ax.set_ylabel("Conversion Rate (%)")
ax.set_title("Daily Conversion Rate by Group")
plt.xticks(rotation=45)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)



# Hypothesis testing

st.subheader("Statistical Hypothesis Testing")

st.write(
    "**Null hypothesis (H₀):** The control and "
    "treatment population conversion rates are equal."
)

st.write(
    "**Alternative hypothesis (H₁):** The population "
    "conversion rates are different."
)

alpha = 0.05
p_value = tests["p_value"]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Z-statistic",
    f'{tests["z_stat"]:.4f}',
)

col2.metric(
    "P-value",
    f"{p_value:.6f}",
)

col3.metric(
    "Significance Level (α)",
    f"{alpha:.2f}",
)

if p_value < alpha:
    st.success(
        "Reject H₀: the data provide evidence of a "
        "difference between the conversion rates."
    )
else:
    st.info(
        "Fail to reject H₀: the data do not provide "
        "sufficient evidence of a difference."
    )



st.subheader("Chi-square Test Cross-check")

st.write(
    f'Chi-square statistic: {tests["chi2"]:.4f}'
)

st.write(
    f'Chi-square p-value: {tests["chi_p"]:.6f}'
)

st.caption(
    "The two-proportion z-test and Chi-square test examine "
    "the same relationship in this two-group, binary-outcome "
    "setting. Their results should generally agree closely "
    "when standard assumptions hold."
)



st.subheader("95% Confidence Interval")

st.write(
    "Estimated difference (treatment − control): "
    f"{absolute_lift * 100:+.2f} percentage points"
)

st.write(
    "95% confidence interval: "
    f"{ci_low * 100:+.2f} to "
    f"{ci_high * 100:+.2f} percentage points"
)

if ci_low > 0:
    st.write(
        "The interval is entirely above zero, indicating "
        "a positive treatment difference under this method."
    )
elif ci_high < 0:
    st.write(
        "The interval is entirely below zero, indicating "
        "a negative treatment difference under this method."
    )
else:
    st.write(
        "The interval includes zero, so it does not establish "
        "a clear positive or negative difference."
    )



st.subheader("Business Interpretation")

if absolute_lift > 0:
    direction = "higher"
elif absolute_lift < 0:
    direction = "lower"
else:
    direction = "the same as"

st.write(
    f"The treatment group's observed conversion rate is "
    f"{direction} the control group's rate."
)

st.write(
    f"The observed absolute difference is "
    f"{absolute_lift * 100:+.2f} percentage points."
)

if p_value < alpha and ci_low > 0:
    st.success(
        "The new page shows a statistically significant "
        "positive conversion-rate difference. Consider the "
        "effect size, costs, and experiment quality before launch."
    )

elif p_value < alpha and ci_high < 0:
    st.warning(
        "The new page shows a statistically significant "
        "negative conversion-rate difference. Investigate "
        "before considering a launch."
    )

else:
    st.info(
        "The experiment does not establish a statistically "
        "significant difference at the 5% level. This does "
        "not prove that both pages perform identically."
    )


st.divider()

st.caption(
    "Dataset: AB Testing Conversion Rate. "
    "The outcome analyzed is conversion, not revenue or profit. "
    "Results depend on the dataset's quality, experiment design, "
    "and the cleaning assumptions used."
)
