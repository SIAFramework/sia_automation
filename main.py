"""
Created on Sat Dec 21 15:10:49 2019
Last Modified on Wed Jan 21 20:40:00 2020
@author: Yashwanth Ramachandra
"""

"""
Standard Libraries
"""

import pandas as pd
import logging
import configparser
import os
import sys
import time
import copy
import subprocess
import signal
import warnings
from nltk.corpus import stopwords
import urllib
import easygui

warnings.filterwarnings("ignore")

"""
Third Party Libraries
"""
from selenium import webdriver
import demoji
import spacy
from spacy_langdetect import LanguageDetector
from pycorenlp import StanfordCoreNLP
import stanfordnlp


"""
Custom Libraries
"""
from scrapers import twscraper, fbscraper, amzscraper, fbcomments
from common import preprocReviews
from features import sentiments, themes, emotions, cluster
from visualization import visualization as viz


"""
Config file
"""
def config_ini():
    config_ini_file = configparser.ConfigParser()
    config_ini_file.read("config.ini")

    return config_ini_file


"""
Main function
"""
def main():
    source = input("Enter the Source: ")
    sources = ["amazon", "facebook", "twitter"]

    if source not in sources:
        logger.error("Please enter the source as either <amazon> or <facebook> or <twitter>")
        print("Please enter the source as either <amazon> or <facebook> or <twitter>")
        raise SystemExit(sys.exit(1))
        
    #phases - function to validate the input the rerun status
    def checkInputStatus(inputoption):
        options=[0,1]
        if inputoption not in options:
            print("Please enter the option as either 0 or 1")
            raise SystemExit(sys.exit(1))
        return 1

    #phases - Get the rerun status from user
    external_data_flag = int(input("Want to upload the data externally?: "))
    checkInputStatus(external_data_flag)
    rerun = int(input("Enter the Re-run status 0/1: "))
    checkInputStatus(rerun)
    #phases - Get the processing options status    
    print("\n---------------- Enter the processing options ---------------- ")
    scrape = None
    if external_data_flag != 1:
        scrape = int(input("\nDo you want to process Scraping 0/1: "))
        checkInputStatus(scrape)
    preproc = int(input("\nDo you want to process Pre processing  0/1: "))
    checkInputStatus(preproc)
    feature = int(input("\nDo you want to process Feature Extraction 0/1: "))
    checkInputStatus(feature)
    clustering = int(input("\nDo you want to process Clustering 0/1: "))
    checkInputStatus(clustering)
    visual = int(input("\nDo you want to process Visualization 0/1: "))
    checkInputStatus(visual)
    
    #phases - Validate the processing options status
    if external_data_flag != 1:
        processoption=str(scrape)+str(preproc)+str(feature)+str(clustering)+str(visual)
    else:
        processoption = str(preproc) + str(feature) + str(clustering) + str(visual)

    if processoption in ['00000']:
        print("\n Not proceeding with any processing------------ END ")
        raise SystemExit(sys.exit(1))
    if processoption in ['11111'] and rerun in [1]:
        print("\n As rerun option is 1, cannot execute all processing------------ END ")
        raise SystemExit(sys.exit(1))
    
    #phases - function to denote the END of processing
    def endprocess():
        logger.info("Total elapsed time: {0}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
        logger.info("End...!!!")
        raise SystemExit(sys.exit(1))

    config = config_ini()

    """
    Pre-requisites
    """
    global keyword, data_tw_post, tw_data_pped, nlp_server, tw_data_emotions, tw_data_clustering
    driver = None
    if not source == "twitter":
        if external_data_flag != 1:
            driver = webdriver.Chrome(executable_path=config['PATHS']['CHROME_DRIVER'])
        demoji.download_codes()

    stanfordnlp_loc = config['PATHS']['SUPPORTING_FILES'] + '\\stanford-corenlp-full-2018-10-05' +"\\"
    cmd = "java -mx4g -cp " + '"*"' + " edu.stanford.nlp.pipeline.StanfordCoreNLPServer"

    nlp_server = subprocess.Popen(cmd, cwd=stanfordnlp_loc)
    spacy_nlp = spacy.load('en_core_web_sm')
    spacy_nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
    sentiment_nlp = StanfordCoreNLP('http://localhost:9000')
    if not os.path.isdir(config['PATHS']['SUPPORTING_FILES'] + '\\en_ewt_models'):
        stanfordnlp.download('en', resource_dir=config['PATHS']['SUPPORTING_FILES'])
    nlp = stanfordnlp.Pipeline(models_dir=config['PATHS']['SUPPORTING_FILES'])

    if source == "twitter":
        try:
            """
            Data Scrapping
            """
            if external_data_flag != 1:
                keyword = input("Enter the Keyword: ")
                if scrape in [1]:
                    pages = input("Enter the no. of Pages: ")
                    logger.info("---------------- Scrapping is Initiated. Please wait...!!! ----------------")
                    data_tw, keyword = twscraper.tw_scraper(keyword, pages)
                    
                    data_tw_post = pd.DataFrame(data_tw)
                    data_tw_post['keyword'] = keyword

                    logger.info("Exporting Scraped data as .csv into Output path. Please wait...!!!")
                    data_tw_post.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\tw_input_data_" + keyword + ".csv",
                                        index=False)
                    logger.info("--------------------- Scraping is Completed...!!! -------------------------")
                    if processoption in ['10000']:
                        endprocess()
                
            """
            Data Pre-Processing
            """
            if preproc in [1]:
                logger.info("------------- Data Pre-processing is Initiated. Please wait...!!! ---------")
                if scrape not in [1]:
                    try:
                        if external_data_flag != 1:
                            data_tw_post=pd.read_csv(config['PATHS']['BASEDIR'] + "\\outputs\\tw_input_data_" + keyword + ".csv")
                        else:
                            keyword = input("Enter the Keyword: ")
                            file_upload = easygui.fileopenbox()
                            data_tw_post = pd.read_csv(file_upload)
                    except Exception as e:
                        print("\n Scraping output file is not available at the mentioned path")
                        logger.error("Exception: {}".format(e))
                tw_data_pp = preprocReviews.twitterPreProcess(data_tw_post, spacy_nlp)
                tw_data_pped = preprocReviews.create_final_input(tw_data_pp, demoji)

                logger.info("Exporting Preprocessing data as .csv into Output path. Please wait...!!!")
                tw_data_pped.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\tw_data_preprocessed_" + keyword + ".csv",
                                index=False)
                logger.info("---------------- Data Pre-processing is Completed...!!! -------------------")
                if processoption[-3:] in ['000']:
                    endprocess()

            """
            Features Extraction: Sentiments
            """
            if feature in [1]:
                logger.info("-------------------------- Features Extraction  ----------------------------")
                logger.info("Sentiment Extraction is in Progress. Please wait...!!!")
                if preproc not in [1]:
                    try:
                        tw_data_pped=pd.read_csv(config['PATHS']['BASEDIR'] + "\\outputs\\tw_data_preprocessed_" + keyword + ".csv")
                    except Exception as e:
                        print("\n Data processing output file is not available at the mentioned path")
                        logger.error("Exception: {}".format(e))
                
                tw_data_sentiment = copy.deepcopy(tw_data_pped)
                tw_data_sentiment['sentiment_new'] = tw_data_sentiment['sentence'].apply(
                    lambda x: sentiments.extract_sentiment(x, sentiment_nlp))
                logger.info("Exporting Sentiments data as .csv into Output path. Please wait...!!!")
                tw_data_sentiment.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\tw_data_sentiments_" + keyword + ".csv",
                                     index=False)
                logger.info("Sentiments Extraction is Completed...!!!")
                

                """
                Features Extraction: Themes
                """
                logger.info("Themes Extraction is in Progress. Please wait...!!!")
                tw_data_themes = copy.deepcopy(tw_data_sentiment)
                tw_data_themes = themes.tag_themes(tw_data_themes, spacy_nlp, nlp)

                logger.info("Exporting as csv into Output Path. Please wait...!!!")
                tw_data_themes.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\tw_data_themes_" + keyword + ".csv",
                                  index=False)
                logger.info("Themes Extraction is Completed...!!!")

                """
                Features Extraction: Emotions
                """
                logger.info("Emotions Extraction is in Progress. Please wait...!!!")
                english_stopwords = stopwords.words('english')
                tw_data_emotions = emotions.tag_emotions(tw_data_themes, english_stopwords, nlp)

                logger.info("Emotions Extraction is Completed...!!!")
                logger.info("Exporting as csv into Output Path. Please wait...!!!")
                tw_data_emotions.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\tw_data_emotions_" + keyword + ".csv",
                                    index=False)
                logger.info("------------------ Features Extraction is Completed...!!! -----------------")
                if processoption[-2:] in ['00']:
                    endprocess()
                    
            nlp_server.kill()

            """
            Features Extraction: Clustering
            """
            if clustering in [1]:
                logger.info("--------------- Clustering is in Progress. Please wait...!!! ---------------")
                themes_map_data = pd.read_csv(config['PATHS']['BASEDIR'] + "\\common_files\\theme_mapping.csv",
                                          error_bad_lines=False, encoding='ISO-8859-1')
                emotions_map_data = pd.read_csv(config['PATHS']['BASEDIR'] + "\\common_files\\emotion_mapping.csv",
                                            error_bad_lines=False, encoding='ISO-8859-1')
                if feature not in [1]:
                    try:
                        tw_data_emotions = pd.read_csv(config['PATHS']['BASEDIR'] + "\\outputs\\tw_data_emotions_" + keyword + ".csv",
                                          error_bad_lines=False, encoding='ISO-8859-1')
                    except Exception as e:
                        print("\n Emotion extraction output file is not available at the mentioned path")
                        logger.error("Exception: {}".format(e))
                    
                tw_data_clustering = cluster.cluster_theme_keywords(tw_data_emotions, themes_map_data)
                tw_data_clustering = cluster.cluster_emotion_keywords(tw_data_clustering, emotions_map_data)
                logger.info("Exporting as csv into Output Path. Please wait...!!!")
                tw_data_clustering.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\tw_data_clustering_" + keyword + ".csv",
                                      index=False)
                logger.info("-------------------- Clustering is Completed...!!! --------------------------")
                if processoption[-1:] in ['0']:
                    endprocess()

            """
            Visualization
            """
            if visual in [1]:
                logger.info("------------------- Visualization is Initiated. Please wait...!!! -----------")
                features = ["themes_keyword", "emotion_keyword", "theme_groups", "emotion_groups"]
                feature_groups = ["theme_groups", "emotion_groups"]
                feature1 = feature_groups[0]
                feature2 = feature_groups[1]

                brand = keyword
                if clustering not in [1]:
                    try:
                        tw_data_clustering = pd.read_csv(config['PATHS']['BASEDIR'] + "\\outputs\\tw_data_clustering_" + keyword + ".csv",
                                            error_bad_lines=False, encoding='ISO-8859-1')
                    except Exception as e:
                        print("\n Clustering output file is not available at the mentioned path")
                        logger.error("Exception: {}".format(e))
                    
                viz_data = copy.deepcopy(tw_data_clustering)
                viz_data = viz_data.loc[viz_data[feature1].notnull(), :]
                viz_data.index = range(len(viz_data))

                viz.plotWordCloud(config=config, source=source, data=viz_data, brand=brand, features=features)
                #viz.frequencyBubblePlot(config=config, source=source, data=viz_data, brand=brand, features=features)
                viz.fitModelAndDraw(config=config, source=source, data=viz_data, __title__=feature1 + 'v/s' + feature2,
                                brand=brand, feature1=feature1, feature2=feature2)
                viz.contigencyTable(config=config, source=source, data=viz_data, brand=brand, feature1=feature1, feature2=feature2)
                viz.frequencyDistribution(config=config, source=source, features=features, data=viz_data, brand=brand)

                logger.info("-------------------- Visualization Completed...!!! --------------------------")
                logger.info("Total elapsed time: {0}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
                logger.info("End...!!!")

        except Exception as e:
            nlp_server.kill()
            logger.error("Exception: {}".format(e))

    elif source == "facebook":
        try:
            """
            Data Scrapping
            """
            if external_data_flag != 1:
                keyword = input("Enter the Keyword: ")
                if scrape in [1]:
                    pages = input("Enter the no. of Pages: ")
                    logger.info("---------------- Scrapping is Initiated. Please wait...!!! ----------------")
                    data_fb, keyword = fbscraper.fb_scraper(keyword, pages)

                    data_fb_post = pd.DataFrame(data_fb)
                    data_fb_post['keyword'] = keyword

                    time.sleep(2)
                    fbpostforlogin = data_fb_post.iloc[:1, ]['post_url'].values[0]

                    driver.get(fbpostforlogin)
                    LogInButton = driver.find_element_by_xpath("//a[@role = 'button']")
                    LogInButton.click()
                    username = driver.find_element_by_id("m_login_email")
                    username.clear()
                    username.send_keys(int(config['FB_LOGINS']['CONTACTNO']))
                    password = driver.find_element_by_id("m_login_password")
                    password.clear()
                    password.send_keys(config['FB_LOGINS']['PASSWORD'])
                    driver.find_element_by_name("login").click()

                    time.sleep(7)
                    fbpostforcomments = copy.deepcopy(data_fb_post)

                    fbpostforcomments = fbpostforcomments.dropna(subset=['post_url'])
                    fb_comments = fbpostforcomments['post_url'].apply(lambda x: fbcomments.scrapeFbComments(x, driver))
                    fb_reviews = pd.concat([r for r in fb_comments], ignore_index=True)

                    fb_reviews = pd.merge(fb_reviews, data_fb_post, left_on='post', right_on='post_url', how="left")
                    fb_reviews = fb_reviews.drop_duplicates(subset='commentWithAuthorname')

                    logger.info("--------------------- Scraping is Completed...!!! -------------------------")
                    logger.info("Exporting as csv into Output Path. Please wait...!!!")
                    fb_reviews.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\fb_input_data_" + keyword + ".csv", index=False)
                    if processoption in ['10000']:
                        endprocess()

            """
            Data Pre-Processing
            """
            if preproc in [1]:
                logger.info("------------- Data Pre-processing is Initiated. Please wait...!!! ---------")
                if scrape not in [1]:
                    try:
                        if external_data_flag != 1:
                            fb_reviews=pd.read_csv(config['PATHS']['BASEDIR'] + "\\outputs\\fb_input_data_" + keyword + ".csv")
                        else:
                            keyword = input("Enter the Keyword: ")
                            file_upload = easygui.fileopenbox()
                            fb_reviews = pd.read_csv(file_upload)
                    except Exception as e:
                        print("\n Scraping output file is not available at the mentioned path")
                        logger.error("Exception: {}".format(e))
                
                fb_data_pp = preprocReviews.fbPreProcess(fb_reviews, spacy_nlp)
                fb_data_pped = preprocReviews.create_final_input(fb_data_pp, demoji)
            
                logger.info("---------------- Data Pre-processing is Completed...!!! -------------------")
                logger.info("Exporting as csv into Output Path. Please wait...!!!")
                fb_data_pped.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\fb_data_preprocessed_" + keyword + ".csv",
                                index=False)
                if processoption[-3:] in ['000']:
                    endprocess()

            """
            Features Extraction: Sentiments
            """
            if feature in [1]:
                logger.info("-------------------------- Features Extraction  ----------------------------")
                logger.info("Sentiments Extraction is in Progress. Please wait..!!!")
                if preproc not in [1]:
                    try:
                        fb_data_pped = pd.read_csv(config['PATHS']['BASEDIR'] + "\\outputs\\fb_data_preprocessed_" + keyword + ".csv")
                    except Exception as e:
                        print("\n Data processing output file is not available at the mentioned path")
                        logger.error("Exception: {}".format(e))
                fb_data_sentiment = copy.deepcopy(fb_data_pped)
                fb_data_sentiment['sentiment_new'] = fb_data_sentiment['sentence'].apply(
                lambda x: sentiments.extract_sentiment(x, sentiment_nlp))
                logger.info("Sentiments Extraction is Completed...!!!")

                logger.info("Exporting as csv into Output Path. Please wait...!!!")
                fb_data_sentiment.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\fb_data_sentiments_" + keyword + ".csv",
                                     index=False)

                """
                Features Extraction: Themes
                """
                logger.info("Themes Extraction is in Progress. Please wait..!!!")

                fb_data_themes = copy.deepcopy(fb_data_sentiment)
                fb_data_themes = themes.tag_themes(fb_data_themes, spacy_nlp, nlp)
                logger.info("Themes Extraction is Completed...!!!")

                logger.info("Exporting as csv into Output Path. Please wait...!!!")
                fb_data_themes.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\fb_data_themes_" + keyword + ".csv",
                                  index=False)

                """
                Features Extraction: Emotions
                """
                logger.info("Emotions Extraction is in Progress. Please wait..!!!")

                english_stopwords = stopwords.words('english')
                fb_data_emotions = emotions.tag_emotions(fb_data_themes, english_stopwords, nlp)
            
                logger.info("Emotions Extraction is Completed...!!!")
                logger.info("Exporting as csv into Output Path. Please wait...!!!")
                fb_data_emotions.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\fb_data_emotions_" + keyword + ".csv",
                                    index=False)

                logger.info("------------------ Features Extraction is Completed...!!! -----------------")
                if processoption[-2:] in ['00']:
                    endprocess()
            nlp_server.kill()

            """
            Features Extraction: Clustering
            """
            if clustering in [1]:
                logger.info("--------------- Clustering is in Progress. Please wait...!!! ---------------")
                themes_map_data = pd.read_csv(config['PATHS']['BASEDIR'] + "\\common_files\\theme_mapping.csv",
                                          error_bad_lines=False, encoding='ISO-8859-1')
                emotions_map_data = pd.read_csv(config['PATHS']['BASEDIR'] + "\\common_files\\emotion_mapping.csv",
                                            error_bad_lines=False, encoding='ISO-8859-1')
                if feature not in [1]:
                    try:
                        fb_data_emotions = pd.read_csv(config['PATHS']['BASEDIR'] + "\\outputs\\fb_data_emotions_" + keyword + ".csv")
                    except Exception as e:
                        print("\n Emotion extraction output file is not available at the mentioned path")
                        logger.error("Exception: {}".format(e))
                fb_data_clustering = cluster.cluster_theme_keywords(fb_data_emotions, themes_map_data)
                fb_data_clustering = cluster.cluster_emotion_keywords(fb_data_clustering, emotions_map_data)
                logger.info("Clustering is Completed...!!!")

                logger.info("Exporting as csv into Output Path. Please wait...!!!")
                fb_data_clustering.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\fb_data_clustering_" + keyword + ".csv",
                                      index=False)
                logger.info("-------------------- Clustering is Completed...!!! --------------------------")
                if processoption[-1:] in ['0']:
                    endprocess()

            """
            Visualization
            """
            if visual in [1]:
                logger.info("------------------- Visualization is Initiated. Please wait...!!! -----------")
            
                features = ["themes_keyword", "emotion_keyword", "theme_groups", "emotion_groups"]
                feature_groups = ["theme_groups", "emotion_groups"]
                feature1 = feature_groups[0]
                feature2 = feature_groups[1]

                brand = keyword
                if clustering not in [1]:
                    try:
                        fb_data_clustering = pd.read_csv(config['PATHS']['BASEDIR'] + "\\outputs\\fb_data_clustering_" + keyword + ".csv")
                    except Exception as e:
                        print("\n Clustering output file is not available at the mentioned path")
                        logger.error("Exception: {}".format(e))
                viz_data = copy.deepcopy(fb_data_clustering)
                viz_data = viz_data.loc[viz_data[feature1].notnull(), :]
                viz_data.index = range(len(viz_data))

                viz.plotWordCloud(config=config, source=source, data=viz_data, brand=brand, features=features)
                #viz.frequencyBubblePlot(config=config, source=source, data=viz_data, brand=brand, features=features)
                viz.fitModelAndDraw(config=config, source=source, data=viz_data, __title__=feature1 + 'v/s' + feature2,
                                brand=brand, feature1=feature1, feature2=feature2)
                viz.contigencyTable(config=config, source=source, data=viz_data, brand=brand, feature1=feature1,
                                feature2=feature2)
                viz.frequencyDistribution(config=config, source=source, features=features, data=viz_data, brand=brand)

                logger.info("-------------------- Visualization Completed...!!! --------------------------")
                logger.info("Total elapsed time: {0}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
                logger.info("End...!!!")

        except Exception as e:
            nlp_server.kill()
            logger.error("Exception: {}".format(e))

    elif source == "amazon":
        try:
            """
            Data Scrapping
            """
            if external_data_flag != 1:
                keyword = input("Enter the Keyword: ")
                if scrape in [1]:
                    logger.info("---------------- Scrapping is Initiated. Please wait...!!! ----------------")
                    review_link_df = pd.read_csv(config['PATHS']['BASEDIR'] + "\\common_files\\review_link.csv",
                                             error_bad_lines=False)
                    review_link_df = review_link_df.drop_duplicates(subset='Review_Link_Href')
                    review_link_df = review_link_df.dropna(subset=['Review_Link_Href'], axis=0)

                    review_link_df['linkset'] = review_link_df.apply(amzscraper.create_linkset, axis=1)
                    review_link_df['linkset2'] = review_link_df['linkset'].apply(lambda x: '|'.join(x))
                    all_links_df = review_link_df['linkset2'].str.split("|", expand=True)
                    total_number_of_pages = len(all_links_df.columns)
                    logger.info("Total no. of Review-Links Scraped: {}".format(len(all_links_df)))
                    review_link_df = pd.concat([review_link_df, all_links_df], axis=1)
                    review_link_df = pd.melt(review_link_df,
                                         id_vars=['web-scraper-order', 'web-scraper-start-url', 'Name', 'Review_Link_Href',
                                                  'Review_Count', 'linkset', 'linkset2'],
                                         value_vars=list(range(0, total_number_of_pages)), value_name='Final_link')
                    review_link_df = review_link_df.sort_values(by=['Review_Link_Href', 'variable'], ascending=[True, True])
                    review_link_df1 = review_link_df[review_link_df['Final_link'].isna() == False]

                    list_dataframe = review_link_df1['Final_link'].apply(lambda x: amzscraper.scrap_reviews(x, driver))
                    reviews_df_stacked = pd.concat([r for r in list_dataframe], ignore_index=True)
                    amz_reviews_data = pd.merge(reviews_df_stacked, review_link_df1, left_on='review_link',
                                            right_on='Final_link', how="left")

                    amz_reviews_data = amz_reviews_data.sort_values(by=['Review_Link_Href', 'Final_link'],
                                                                ascending=[True, True])

                    logger.info("--------------------- Scraping is Completed...!!! -------------------------")
                    logger.info("Exporting as csv into Output Path. Please wait...!!!")
                    amz_reviews_data.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\amz_input_data_" + keyword + ".csv",
                                        index=False)
                    if processoption in ['10000']:
                        endprocess()

            """
            Data Pre-Processing
            """
            if preproc in [1]:
                logger.info("------------- Data Pre-processing is Initiated. Please wait...!!! ---------")
                if scrape not in [1]:
                    try:
                        if external_data_flag != 1:
                            amz_reviews_data=pd.read_csv(config['PATHS']['BASEDIR'] + "\\outputs\\amz_input_data_" + keyword + ".csv")
                        else:
                            keyword = input("Enter the Keyword: ")
                            file_upload = easygui.fileopenbox()
                            amz_reviews_data = pd.read_csv(file_upload)
                    except Exception as e:
                        print("\n Scraping output file is not available at the mentioned path")
                        logger.error("Exception: {}".format(e))
                amz_data_pp = preprocReviews.amazonPreProcess(amz_reviews_data, spacy_nlp)
                amz_data_pped = preprocReviews.create_final_input(amz_data_pp, demoji)
            
                logger.info("---------------- Data Pre-processing is Completed...!!! -------------------")
                logger.info("Exporting as csv into Output Path. Please wait...!!!")
                amz_data_pped.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\amz_data_preprocessed_" + keyword + ".csv",
                                 index=False)
                if processoption[-3:] in ['000']:
                    endprocess()

            """
            Features Extraction: Sentiments
            """
            if feature in [1]:
                logger.info("-------------------------- Features Extraction  ----------------------------")
                logger.info("Sentiments Extraction is in Progress. Please wait..!!!")
                if preproc not in [1]:
                    try:
                        amz_data_pped = pd.read_csv(config['PATHS']['BASEDIR'] + "\\outputs\\amz_data_preprocessed_" + keyword + ".csv")
                    except Exception as e:
                        print("\n Data processing output file is not available at the mentioned path")
                        logger.error("Exception: {}".format(e))

                amz_data_sentiment = copy.deepcopy(amz_data_pped)
                amz_data_sentiment['sentiment_new'] = amz_data_sentiment['sentence'].apply(
                lambda x: sentiments.extract_sentiment(x, sentiment_nlp))
                logger.info("Sentiments Extraction is Completed...!!!")

                logger.info("Exporting as csv into Output Path. Please wait...!!!")
                amz_data_sentiment.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\amz_data_sentiments_" + keyword + ".csv",
                                      index=False)

                """
                Features Extraction: Themes
                """
                logger.info("Themes Extraction is in Progress. Please wait..!!!")

                amz_data_themes = copy.deepcopy(amz_data_sentiment)
                amz_data_themes = themes.tag_themes(amz_data_themes, spacy_nlp, nlp)
                logger.info("Themes Extraction is Completed...!!!")

                logger.info("Exporting as csv into Output Path. Please wait...!!!")
                amz_data_themes.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\amz_data_themes_" + keyword + ".csv",
                                   index=False)

                """
                Features Extraction: Emotions
                """
                logger.info("Emotions Extraction is in Progress. Please wait..!!!")

                english_stopwords = stopwords.words('english')
                amz_data_emotions = emotions.tag_emotions(amz_data_themes, english_stopwords, nlp)
                logger.info("Emotions Extraction is Completed...!!!")

                logger.info("Exporting as csv into Output Path. Please wait...!!!")
                amz_data_emotions.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\amz_data_emotions_" + keyword + ".csv",
                                     index=False)

                logger.info("------------------ Features Extraction is Completed...!!! -----------------")
                if processoption[-2:] in ['00']:
                    endprocess()
            nlp_server.kill()

            """
            Features Extraction: Clustering
            """
            if clustering in [1]:
                logger.info("--------------- Clustering is in Progress. Please wait...!!! ---------------")
            
                themes_map_data = pd.read_csv(config['PATHS']['BASEDIR'] + "\\common_files\\theme_mapping.csv",
                                          error_bad_lines=False, encoding='ISO-8859-1')
                emotions_map_data = pd.read_csv(config['PATHS']['BASEDIR'] + "\\common_files\\emotion_mapping.csv",
                                            error_bad_lines=False, encoding='ISO-8859-1')
                if feature not in [1]:
                    try:
                        amz_data_emotions = pd.read_csv(config['PATHS']['BASEDIR'] + "\\outputs\\amz_data_emotions_" + keyword + ".csv")
                    except Exception as e:
                        print("\n Emotion extraction output file is not available at the mentioned path")
                        logger.error("Exception: {}".format(e))
                        
                amz_data_clustering = cluster.cluster_theme_keywords(amz_data_emotions, themes_map_data)
                amz_data_clustering = cluster.cluster_emotion_keywords(amz_data_clustering, emotions_map_data)
                logger.info("-------------------- Clustering is Completed...!!! --------------------------")

                logger.info("Exporting as csv into Output Path. Please wait...!!!")
                amz_data_clustering.to_csv(config['PATHS']['BASEDIR'] + "\\outputs\\amz_data_clustering_" + keyword + ".csv", index=False)
                if processoption[-1:] in ['0']:
                    endprocess()

            """
            Visualization
            """
            if visual in [1]:
                logger.info("------------------- Visualization is Initiated. Please wait...!!! -----------")
            
                features = ["themes_keyword", "emotion_keyword", "theme_groups", "emotion_groups"]
                feature_groups = ["theme_groups", "emotion_groups"]
                feature1 = feature_groups[0]
                feature2 = feature_groups[1]

                brand = keyword
                if clustering not in [1]:
                    try:
                        amz_data_clustering = pd.read_csv(config['PATHS']['BASEDIR'] + "\\outputs\\amz_data_clustering_" + keyword + ".csv")
                    except Exception as e:
                        print("\n Clustering output file is not available at the mentioned path")
                        logger.error("Exception: {}".format(e))
                viz_data = copy.deepcopy(amz_data_clustering)
                viz_data = viz_data.loc[viz_data[feature1].notnull(), :]
                viz_data.index = range(len(viz_data))

                viz.plotWordCloud(config=config, source=source, data=viz_data, brand=brand, features=features)
                #viz.frequencyBubblePlot(config=config, source=source, data=viz_data, brand=brand, features=features)
                viz.fitModelAndDraw(config=config, source=source, data=viz_data, __title__=feature1 + 'v/s' + feature2,
                                brand=brand, feature1=feature1, feature2=feature2)
                viz.contigencyTable(config=config, source=source, data=viz_data, brand=brand, feature1=feature1,
                                feature2=feature2)
                viz.frequencyDistribution(config=config, source=source, features=features, data=viz_data, brand=brand)

                logger.info("-------------------- Visualization Completed...!!! --------------------------")
                logger.info("Total elapsed time: {0}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
                logger.info("End...!!!")

        except Exception as e:
            nlp_server.kill()
            logger.error("Exception: {}".format(e))

    return 1


if __name__ == '__main__':
    start_time = time.time()
    config_log = config_ini()
    print("Please find the Logs here: {}".format(config_log['PATHS']['BASEDIR'] + "\\logs\\"))
    print("Please find the intermediate outputs here: {}".format(config_log['PATHS']['BASEDIR']) + "\\outputs\\")
    logging.basicConfig(filename=config_log['PATHS']['BASEDIR'] + "\\logs\\sia_log.log",
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    main()
