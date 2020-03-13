"""
Created by: Yashwanth Ramachandra
Owned by: Yashwanth Ramachandra
Last Modified: 26-Nov-2019
"""

import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import prince as pr
import altair as alt

import logging
logger = logging.getLogger('sialogger')


"""
Stopwords
"""
def stopwords(brand):

    stopwords_ls = set(STOPWORDS)
    cust_stopwords = ["make","product","NEUTRAL", "schwarz", "fao","schwartz","store", "faoschwarz","chili"]
    stopwords_ls.update(cust_stopwords)

    if brand == "fao":
        stopwords_fao = []
        with open("stopwordsFAO.txt", 'r') as txt:
            for x in txt.readlines():
                stopwords_fao.append(x.split('\n')[0])
        stopwords_ls.update(stopwords_fao)

    rm_stopwords = set(stopwords_ls)
    return rm_stopwords


"""
Word-Cloud
"""
def plotWordCloud(config, source, data, brand, features):
    __stopwords__ = stopwords(brand=brand)

    for feature_i in features:

        dedup = ["source", "sentence"] + [feature_i]
        data_dedup = data.drop_duplicates(subset=dedup, keep='first', inplace=False)

        data_dedup = data_dedup.loc[:, [feature_i]]
        wc_data = data_dedup.query(feature_i + " not in @__stopwords__")

        plt.figure(figsize=(10, 10))
        wordcloud = WordCloud(
            background_color='black',
            max_words=100,
            max_font_size=120,
            random_state=100
        ).generate_from_frequencies(wc_data[feature_i].value_counts())

        # Plotting the word cloud
        logger.info("Plotting WordCloud for {}".format(feature_i))
        plt.imshow(wordcloud)
        plt.title("WORD CLOUD for " + brand + ": " +feature_i.upper(), fontsize = 20)
        plt.axis('off')
        plt.savefig(config['PATHS']['BASEDIR'] + "\\outputs\\"+ source + "_WordCloud_" + brand + "_" + feature_i + ".png")


"""
Bubble Plot
"""
def frequencyBubblePlot(config, source, data, brand, features):

    for feature_i in features:
        data_feature = data.loc[:, ["source",feature_i]]
        data_ft = pd.DataFrame(data_feature[feature_i].value_counts().reset_index())
        data_ft.columns = [feature_i]+["count"]

        logger.info("Plotting Bubble Plot for {}".format(feature_i))
        alt.Chart(data_ft).mark_circle().encode(
            x=alt.X(feature_i, axis=alt.Axis(title=feature_i.split("_")[0].upper())),
            y=alt.Y("count", axis=alt.Axis(title="Frequency")),
            size="count"
        ).interactive().save(config['PATHS']['BASEDIR'] + "\\outputs\\" + source + "_Bubble_Plot_" + brand + "_" + feature_i + ".png")


"""
Correspondance Analysis
"""
def getStructureDF(data, feature1, feature2):
    data.loc[:,'freq'] = 1
    data_ca = None

    if feature1 != feature2:
        data_ca = pd.pivot_table(data, index=[feature1], columns=[feature2],
                                          aggfunc=np.sum, values=['freq'], fill_value=0)
        data_ca.columns = data_ca.columns.droplevel()

    elif feature1 == feature2:
        data[feature1 + str(2)] = data[feature1]
        feature2 = feature2 + str(2)
        data_ca = pd.pivot_table(data, index=[feature1], columns=[feature2],
                                 aggfunc=np.sum, values=['freq'], fill_value=0)
        data_ca.columns = data_ca.columns.droplevel()

    return data_ca



def fitModelAndDraw(config, source, data, __title__, brand, feature1, feature2):
    ca = pr.CA(n_components=2, n_iter=10, copy=True, check_input=True, engine='auto', random_state=42)
    ca_data = getStructureDF(data=data, feature1=feature1, feature2=feature2)

    caObj = ca.fit(ca_data)
    __ax__ = caObj.plot_coordinates(X=ca_data, ax=None, figsize=(20, 12), x_component=0, y_component=1,
                                show_row_labels=True, show_col_labels=True
                                )
    __ax__.axis("off")
    __ax__.axhline(False)
    __ax__.axvline(False)

    def bubblePlotData(data_ca):
        cols_cnt = ["groups", "count"]
        cols = ["groups", "X", "Y", "group_flag"]
        cols_order = ["group_flag", "groups", "X", "Y"]
        theme_coord = caObj.row_coordinates(data_ca).reset_index()
        theme_coord["group_flag"] = "Themes"
        theme_coord.columns = cols
        emotion_coord = caObj.column_coordinates(data_ca).reset_index()
        emotion_coord["group_flag"] = "Emotions"
        emotion_coord.columns = cols

        coord_data = theme_coord.append(emotion_coord)
        coord_data = coord_data[cols_order]

        freq_dist_theme = data[feature1].value_counts().reset_index()
        freq_dist_theme.columns = cols_cnt
        freq_dist_emotion = data[feature2].value_counts().reset_index()
        freq_dist_emotion.columns = cols_cnt
        freq_dist = freq_dist_theme.append(freq_dist_emotion)
        buble_plot_data = coord_data.merge(freq_dist, on="groups", how="left")

        return buble_plot_data

    bubble_plot_data = bubblePlotData(ca_data)
    bubble_plot_data.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\" + source +"_bubble_plot_data_" + brand + ".csv", index=False)

    logger.info("Plotting Corresponding Chart...!!!")
    plt.title(feature1 + ' v/s ' + feature2)
    plt.savefig(config['PATHS']['BASEDIR'] + "\\outputs\\" + source + "_CA_" + brand + "_" +feature1 + "_" + feature2 + ".png")



"""
Contigency Table
"""
def contigencyTable(config, source, data, brand, feature1, feature2):

    logger.info("Drawing Contigency Table...!!!")
    data.loc[:, 'freq'] = 1
    data_ct = pd.pivot_table(data, index=[feature1], columns=[feature2],
                             aggfunc=np.sum, values=['freq'], fill_value=0)
    data_ct.columns = data_ct.columns.droplevel()

    data_ct.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\" + source + "_Contingency_Tab_" + brand + "_" + \
                   feature1 + "_" + feature2 + ".csv", index=True)


"""
Frequency Table
"""
def frequencyDistribution(config, source, features, data, brand):

    __stopwords__ = stopwords(brand=brand)
    freq_dist_data = pd.DataFrame()
    
    for feature_i in features:
        data_feature = data.loc[:, ["source", "sentence", feature_i]]
        dedup = ["source", "sentence"] + [feature_i]
        data_feature = data_feature.drop_duplicates(subset=dedup, keep='first', inplace=False)
        freq_dist = data_feature.loc[:, [feature_i]]
        freq_dist = freq_dist.query(feature_i + " not in @__stopwords__")
        freq_dist = pd.DataFrame(freq_dist[feature_i].value_counts().reset_index())
        freq_dist.columns = ["words","count"]
        freq_dist["features"] = feature_i
        freq_dist_data = freq_dist_data.append(freq_dist, sort=False)
        logger.info("Frequency Distribution is generated for {}".format(feature_i))
    freq_dist_data.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\" + source + "_freq_dist_data_" + brand +".csv", index=False)