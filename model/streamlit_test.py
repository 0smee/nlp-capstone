### import libraries
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go

import source 
st.markdown('# Identifying customer pain points in Amazon Reviews')
# Set up input field with st.text_input()
text = st.text_input("#### 👩‍💻 Enter your review here:", "What a great phone case! So cute and fits perfectly!")


choice = ["logistic-bal-model.pkl", "logistic-unbal-model.pkl", "decision-tree-bal-model.pkl", "decision-tree-unbal-model.pkl", "random-forest-unbal-model.pkl", "random-forest-bal-model.pkl"]
option = st.selectbox('Pick a model:',choice)

# Load the model using joblib
model = joblib.load(option)
st.markdown(f"*Choice: **{option}***")

# Use the model to predict sentiment & save to a variable called prediction
prediciton = model.predict({text})
proba = model.predict_proba({text})[0]

# based on prediction display something to user
if prediciton == 1:
    st.markdown("### Prediction: Positive! 🤗")
    st.markdown(f"#### Confidence: {proba[1]*100:.0f}%")   
else:
    st.markdown("### Prediction: Negative! 😨")
    st.markdown(f"#### Confidence: {proba[0]*100:.0f}%")
transformed_text = model[0].transform([text])
feature_names = model[0].get_feature_names_out()

# if we use a logistic model:
if option in ["logistic-bal-model.pkl", "logistic-unbal-model.pkl"]:
    # contribution calcs
    coefficients = model[1].coef_[0]
    contributions = transformed_text.multiply(coefficients)
    contributions_dense = contributions.toarray().flatten()

    # feature contributions
    contributions_df = pd.DataFrame({
        'Feature': feature_names,  
        'Contribution': contributions_dense
    }).sort_values(by="Contribution",ascending=False)
    contributions_df['Feature'] = contributions_df['Feature'].str.replace(r'^.*gram__', '', regex=True)
    # non-zero contributions of features
    top_n = st.slider('Select number of features to display', min_value=1, max_value=10, value=3)
    non_zero_contributions = contributions_df.loc[contributions_df['Contribution'] != 0]
    top_positive = non_zero_contributions[non_zero_contributions['Contribution'] > 0].nlargest(top_n, 'Contribution')
    top_negative = non_zero_contributions[non_zero_contributions['Contribution'] < 0].nsmallest(top_n, 'Contribution')
    # add most positive and negative
    top = pd.concat([top_positive, top_negative])
    top = top.drop_duplicates()
    colors = ['cornflowerblue' if x > 0 else 'coral' for x in top['Contribution']]
    # https://plotly.com/python/graph-objects/
    fig = go.Figure(data=[go.Bar(
        x=top['Feature'],
        y=top['Contribution'],
        marker_color=colors 
    )])
    # aesthetics ;
    fig.update_layout(title='Most influential words:',
                    xaxis_title='Feature',
                    yaxis_title='Influence', 
                    title_font_color="coral",title_font_size=20)

    fig.update_xaxes(titlefont=dict(size=24)) 
    fig.update_xaxes(tickfont=dict(size=24)) 
    fig.update_yaxes(titlefont=dict(size=24)) 
    # show chart
    st.plotly_chart(fig)

    # feature importance for logistoic (coefficient plot)
    word_counts = pd.DataFrame(
    {"coefficients": model[1].coef_[0]},
    index=feature_names).sort_values("coefficients", ascending=False)

    word_counts2 = pd.DataFrame(
    {"coefficients": model[1].coef_[0]},
    index=feature_names).sort_values("coefficients", ascending=True)
    
    p = word_counts.head(5)
    n = word_counts2.head(5)
    p.columns = ['pos']
    n.columns = ['neg']
    p_and_n = pd.concat([p,n])
    stacked_data = p_and_n.stack().reset_index()
    stacked_data.columns = ['feature', 'type', 'coefficient']
    sorted_data = stacked_data.sort_values(by='coefficient', ascending=False)
    ####
    colors = ['cornflowerblue' if tp == "pos" else 'coral' for tp in sorted_data['type']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_data['feature'].str.replace(r'^.*gram__', '', regex=True),
        y=sorted_data['coefficient'],
        orientation='v',
        marker_color=colors))
    fig.update_layout(
        title_text="We expect these words to have the biggest influence:",
        title_font_size=20,title_font_color="coral",
        xaxis_title="Coefficients",
        yaxis_title="Features",
        xaxis=dict(titlefont=dict(size=24),tickfont=dict(size=20)),
        yaxis=dict(titlefont=dict(size=24),tickfont=dict(size=16)))
    
    st.plotly_chart(fig)
elif option in ["decision-tree-bal-model.pkl", "decision-tree-unbal-model.pkl", "random-forest-unbal-model.pkl", "random-forest-bal-model.pkl"]:
    fi = model[1].feature_importances_

    feature_imp = pd.DataFrame(fi.reshape(1,-1), columns=feature_names, index=['Importances']).T
    sorted_imp = feature_imp.sort_values(by='Importances', ascending=False)
    sorted_imp.index = sorted_imp.index.str.replace(r'^.*gram__', '', regex=True)
    top_m = st.slider('Select number of features to display', min_value=1, max_value=10, value=6)

    # st.dataframe(sorted_imp.head(10))
    fig = go.Figure(data=[go.Bar(
        x=sorted_imp.head(top_m).index,
        y=sorted_imp.head(top_m)['Importances'],
        marker_color="cornflowerblue" 

    )])
    # aesthetics ;
    fig.update_layout(title='We expect these words to have the biggest influence:',
                    title_font_size=20,
                    xaxis_title='Feature',
                    yaxis_title='Influence', 
                    title_font_color="coral")

    fig.update_xaxes(titlefont=dict(size=20)) 
    fig.update_xaxes(tickfont=dict(size=24)) 
    fig.update_yaxes(titlefont=dict(size=20)) 

    st.plotly_chart(fig)
