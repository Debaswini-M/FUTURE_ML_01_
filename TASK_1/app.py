#Importing libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import shap
import joblib


#page configuration
st.set_page_config(
    page_title = "Spotify Satisfaction Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🎧Spotify Satisfaction Prediction ")
st.write("An ML dashboard for predictions ,insights and analysis")


# SPOTIFY GREEN THEME
spotify_css = """
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0b0f0e 0%,      /* deep black-green */
        #0d1312 15%,     /* charcoal */
        #0f1a16 30%,     /* dark forest */
        #1DB954 55%,     /* spotify neon green */
        #158c5d 75%,     /* teal-green */
        #0a3327 100%     /* deep teal */
    ) !important;
    background-attachment: fixed;
}

section[data-testid="stSidebar"] {
    background-color: rgba(10, 20, 15, 0.92) !important;
    border-right: 1px solid #1DB954;
}

</style>
"""

st.markdown(spotify_css, unsafe_allow_html=True)




#navigation bar
st.sidebar.title("Menu")
page = st.sidebar.radio("GO to:",[
    "Upload & Predict",
    "Pie Chart",
    "Confusion Matrix",
    "ROC Curve",
    "Feature Importance(SHAP)",
    "Feature Importance",
    "Overall Insights"
])
st.sidebar.write('---')
st.sidebar.write('Model:**XGBOOST(multi-class)**')
st.sidebar.write('Version:1.0.0')

#Loading the model, test set
@st.cache_resource
def load_model():
    return joblib.load('Model_Pipeline.pkl')

@st.cache_data
def load_test_set():
    try:
        test_df = pd.read_csv('test_with_labels.csv')
        return test_df
    except:
        return None

model = load_model()
test_df = load_test_set()


#page1:upload and predict
if page == "Upload & Predict":
    st.header("Upload & Predict")
    uploaded_file = st.file_uploader("Upload a CSV file",type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())
        #predictions
        preds = model.predict(df)
        probs = model.predict_proba(df)

        output = df.copy()
        output["predicted_label"] = preds
        for i in range(probs.shape[1]):
            output[f"Satisfaction_class{i}"] = probs[:,i]

        st.subheader("Prediction")
        st.dataframe(output.head())

        #storing the predictions so that other pages will use
        st.session_state['last_prediction'] = output

# #Page:2 Pie chart
elif page == "Pie Chart":
    st.header("Pie Chart")
    if "last_prediction" not in st.session_state:
        st.warning("⚠️No predictions found.")
        st.stop()
    else:
        output = st.session_state['last_prediction']

    fig_pie,ax_pie = plt.subplots(figsize=(6,6))
    counts = output["predicted_label"].value_counts()
    ax_pie.pie(counts,labels=counts.index,autopct='%1.1f%%',startangle=90)
    ax_pie.axis('equal')
    plt.style.use('dark_background')
    st.pyplot(fig_pie)
    st.success("Pie Chart Generated Successfully!!!")


#page:3 confusion matrix
elif page == "Confusion Matrix":
    st.header("Confusion Matrix")

    if test_df is None:
        st.error("Test set not found")
        st.stop()
    elif 'last_prediction' not in st.session_state:
        st.warning("No predictions found.")
        st.stop()

    else:
        X_test = test_df.drop(columns=["True_Label"])
        y_true = test_df['True_Label']
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        plt.style.use('dark_background')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Class 0","Class 1","Class 2"])
        disp.plot(ax=ax,cmap="gnuplot2_r",colorbar=True)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        st.pyplot(fig)

#page:4 ROC CURVE
elif page == "ROC Curve":
    st.header("ROC Curve(Class wise)")

    if test_df is None:
        st.error("Test set not found")
        st.stop()
    elif 'last_prediction' not in st.session_state:
        st.warning("No predictions found.")
        st.stop()
    else:
        X_test = test_df.drop(columns=["True_Label"])
        y_true = test_df["True_Label"]
        probs = model.predict_proba(X_test)
        fig,ax = plt.subplots(figsize=(6,5))

        for i in range(probs.shape[1]):
            fpr,tpr,_ = roc_curve(y_true==i, probs[:,i])
            roc_auc = auc(fpr,tpr)
            plt.plot(fpr,tpr,label=f"Class{i} (AUC={roc_auc:.2f})")

        plt.style.use('dark_background')
        ax.plot([0,1],[0,1],'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")

        st.pyplot(fig)

#page:5 Feature Importance SHAP Plot

elif page == "Feature Importance(SHAP)":
    st.header("🔎Feature Importance(SHAP)")
    if test_df is None:
        st.warning("Test set not found")
        st.stop()
    elif 'last_prediction' not in st.session_state:
        st.warning("No predictions found.")
        st.stop()
    else:
        X_sample = test_df.drop(columns=["True_Label"]).iloc[:100]
        prep = model.named_steps['prep']
        clf = model.named_steps['clf']

        X_transformed = prep.transform(X_sample)
        feature_names = prep.get_feature_names_out()

        explainer = shap.TreeExplainer(clf.get_booster())
        shap_vals = explainer.shap_values(X_transformed)

        class_values = shap_vals[1] if isinstance(shap_vals,list) else shap_vals
        fig= plt.figure(figsize=(10,8))
        shap.summary_plot(class_values,X_transformed,feature_names=feature_names,plot_type="bar",show=False)
        plt.legend(loc="best")
        st.pyplot(fig)


#Page:6 Feature Importance

elif page == "Feature Importance":
    st.header("XGBOOST Feature Importance")
    if 'last_prediction' not in st.session_state:
        st.warning("No predictions found.")
        st.stop()

    clf = model.named_steps['clf']
    feature_names = model.named_steps['prep'].get_feature_names_out()
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)

    st.dataframe(importance.head(20))
    plt.style.use('dark_background')
    plt.figure(figsize=(10,8))
    sns.barplot(data= importance.head(15),x="importance", y="feature")
    plt.show()



#PAGE:7 OVERALL BUSINESS INSIGHTS

elif page == "Overall Insights":
    st.header("Overall Insights")


    if 'last_prediction' not in st.session_state:
        st.warning("No predictions found.")
        st.stop()
    else:
        st.markdown("""
### High Satisfaction Users(class2)
- Highest retention potential
- Should be nudged towards premium features
- Highly engaged audience
### Neutral Users(class1)
- Largest segment
- Easy to convert to high satisfaction
- Recommend targeting with personalized suggestions
### Low Satisfaction Users(class0)
- At risk of churn
- Need follow-up campaigns, curated playlists, UX improvements

---
 
### Key Model Learnings (Based On SHAP and Moodel Patterns)

#### 1. Music Recommendation Rating = The #1 Driver of Satisfaction
Improving recommendation accuracy directly increases user happiness.
            
#### 2. Podcast Behavior Strongly Predicts Satisfaction drive satisfaction more than expected.
Features like:
- Podcast duration  
- Host preference  
- Frequency  

#### 3.  Device Consistency Matters
-  Users sticking to smartphones show higher satisfaction.
-  Frequent device-switchers report lower satisfaction.
#### 4. Mood-Based Listening Is Underutilized
-Users influenced by **Relaxation/Stress Relief** respond well to mood-matching recommendations.

---
## Summary Table
| Recommendation                               |   Impact    |    Effort     |
|----------------------------------------------|-------------|---------------|
| Improve recommendation engine accuracy       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐       |
| Build mood-based adaptive playlist engine    | ⭐⭐⭐⭐   | ⭐⭐⭐⭐     |
| Introduce micro premium trials               | ⭐⭐⭐⭐   | ⭐            |
| Boost long-form podcast creators             | ⭐⭐⭐     | ⭐⭐⭐        |
| Fix UX gaps for multi-device listening       | ⭐⭐⭐     | ⭐⭐          |

---

## ✨ Pro Insight
Low-satisfaction users show **fragmented listening sessions** due to:
- Frequent device switching  
- Shorter content  

This suggests UX improvements in **cross-device handoff** can meaningfully increase satisfaction.***

---
""")