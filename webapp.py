import pandas as pd
import streamlit as st
from insert_reports import get_project

st.title("Visualise your skore Project")


@st.cache_resource
def load_project():
    return get_project()


local_project = load_project()

project_summary = local_project.summarize()
st.dataframe(project_summary, use_container_width=True)

report_ids = [report[1] for report in local_project.summarize().index.tolist()]
selected_report_id = st.selectbox("Select Report by ID", report_ids)

report = local_project.get(selected_report_id)
df = report.metrics.summarize().frame()
ml_task = project_summary.xs(selected_report_id, level='id')['ml_task'].iloc[0]

report_details = pd.DataFrame(project_summary.xs(selected_report_id, level='id'))
report_details_t = report_details.astype(str).T
report_details_t.columns = ["Value"]
st.write("### Report details")
st.dataframe(report_details_t, use_container_width=True)

if ml_task == "regression":

    show_metrics = st.checkbox("📊 Show metrics summary", value=True)
    if show_metrics:
        st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_prediction_error = st.checkbox("🎯 Prediction error plot")
    with col2:
        show_actual_vs_predicted = st.checkbox("📈 Actual vs Predicted plot")

    if show_prediction_error:
        disp = report.metrics.prediction_error()
        _ = disp.plot()
        st.pyplot(disp.figure_)

    if show_actual_vs_predicted:
        disp = report.metrics.prediction_error()
        _ = disp.plot(kind="actual_vs_predicted")
        st.pyplot(disp.figure_)

elif ml_task == "binary-classification":
    df.index = df.index.set_levels(
        df.index.levels[1].map(lambda x: None if x == '' else x), level=1
    )

    st.subheader("Binary Classification Visualisations")

    show_metrics = st.checkbox("Show metrics summary", value=True)
    if show_metrics:
        st.dataframe(df, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        show_confusion_matrix = st.checkbox("Confusion Matrix")
    with col2:
        show_roc_curve = st.checkbox("ROC Curve")
    with col3:
        show_precision_recall = st.checkbox("Precision-Recall Curve")

    if show_confusion_matrix:
        fig = report.metrics.confusion_matrix().plot()
        st.pyplot(fig.figure_)

    if show_roc_curve:
        disp = report.metrics.roc()
        _ = disp.plot()
        st.pyplot(disp.figure_)

    if show_precision_recall:
        disp = report.metrics.precision_recall()
        _ = disp.plot()
        st.pyplot(disp.figure_)

else:
    st.warning(f"Unsupported ML task: {ml_task}")
    st.info("Currently supported tasks: regression, binary-classification")

