# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:20:27 2020

@author: AkOjha
"""

from selenium import webdriver
import pandas as pd
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import copy
import re
import logging
logger = logging.getLogger('scraperlogger')

#review_link_df1 = getreview_link("lakme")

def findnum(x):
    if len(re.findall(r'\d+', x))==0:
        return 0
    else:
        return int(re.findall(r'\d+', x)[0])


def getreview_link(keyword):
    search_term = keyword
    print("Search term =",search_term)

    driver = webdriver.Chrome(r'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe')
    driver.get("https://www.amazon.com")  
    driver.refresh()

    searchTextBox=driver.find_element_by_id("twotabsearchtextbox")
    searchTextBox.clear()
    searchTextBox.send_keys(search_term.lower())
    searchTextBox.send_keys(Keys.RETURN)

    pagination_text = driver.find_elements_by_xpath("//ul[@class='a-pagination']")[0].text

    list_pages = re.findall(r'\d+', pagination_text)
    list_pages = [int(i) for i in list_pages]
    
    num_of_pages = max(list_pages)
    final_data = []
    print("no.of pages = ",num_of_pages)
    for n in range(0, num_of_pages):
    
        ids = driver.find_elements_by_xpath("//a[@class='a-link-normal a-text-normal']")
        number_of_prod = len(ids)
        print("No.of products = ",number_of_prod)
        brand = []
        product_name = []
        num_of_reviews = []
        price = []
        review_links = []
        review_id = []
        
        for i in range(0,number_of_prod):
            try:
        
                driver.find_elements_by_xpath("//a[@class='a-link-normal a-text-normal']")[i].click()
                 
                try:
                    brand_element = driver.find_elements_by_xpath("//div[@id='brandBar_feature_div']")[0]
                    brandtemp = brand_element.text
                    brand.append(brandtemp)
                except Exception as e:
                    logger.info("Exception is {}".format(e))
                    brand.append(None)
                print("brand is : ",brand)
                try:
                    product_name_element = driver.find_elements_by_xpath("//span[@id='productTitle']")[0]
                    product_nametemp = product_name_element.text
                    product_name.append(product_nametemp)
                except Exception as e:
                    logger.info("Exception is {}".format(e))
                    product_name.append(None)
    
                try:
                    price_element = driver.find_elements_by_xpath("//span[@id='priceblock_ourprice']")[0]
                    pricetemp = price_element.text
                    price.append(pricetemp)
                except Exception as e:
                    logger.info("Exception is {}".format(e))
                    price.append(None)
    
                try:
                    num_of_reviews_element = driver.find_elements_by_xpath("//a[@data-hook='see-all-reviews-link-foot']")[0]
                    num_of_reviewstemp = num_of_reviews_element.text
                    num_of_reviews.append(num_of_reviewstemp)
                except Exception as e:
                    logger.info("Exception is {}".format(e))
                    num_of_reviews.append(None)
            
                try:
                    review_links_element = driver.find_elements_by_xpath("//a[@data-hook='see-all-reviews-link-foot']")[0]
                    review_linksstemp = review_links_element.get_attribute("href")
                    review_links.append(review_linksstemp)
                except Exception as e:
                    logger.info("Exception is {}".format(e))
                    review_links.append(None)
            except Exception as e:
                driver.back()
                driver.refresh()
                continue            
            
    
        temp_review_data = pd.DataFrame({'brand':brand, 'product_name':product_name, 'price':price, 'num_of_reviews':num_of_reviews, 'review_links':review_links})
        temp_review_data['review_id'] = list(range(0, temp_review_data.shape[0]))
        print("temp_review_data is : ",temp_review_data)
    

    final_data.append(temp_review_data)
    final_data1 = pd.concat(final_data)    
    print("final_data1 is : ",final_data1)

    pos_reviews = []
    neg_reviews = []
    review_id = []

    driver = webdriver.Chrome(r'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe')

    for i in range(0,final_data1.shape[0]):
        review_id.append(i)
        if final_data1.iloc[i,]['review_links'] == None:
            pos_reviews.append(0)
            neg_reviews.append(0)
            continue
        else:
            driver.get(final_data1.iloc[i,]['review_links'])
            try:    
                num_of_pos_reviews_element = driver.find_elements_by_xpath("//a[@data-reftag='cm_cr_arp_d_viewpnt_lft']")[0]
                num_of_pos_reviews = num_of_pos_reviews_element.text
                final_data1.iloc[i,]['pos_reviews'] = num_of_pos_reviews
                pos_reviews.append(num_of_pos_reviews)
            except Exception as e:
                logger.info("Exception is {}".format(e))
                pos_reviews.append(None)
            try:        
                num_of_neg_reviews_element = driver.find_elements_by_xpath("//a[@data-reftag='cm_cr_arp_d_viewpnt_rgt']")[0]
                num_of_neg_reviews = num_of_neg_reviews_element.text
                final_data1.iloc[i,]['neg_reviews'] = num_of_neg_reviews
                neg_reviews.append(num_of_neg_reviews)
            except Exception as e:
                logger.info("Exception is {}".format(e))
                neg_reviews.append(None)
        
    num_reviewsDF = pd.DataFrame({'review_id':review_id, 'pos_reviews':pos_reviews, 'neg_reviews':neg_reviews})


    num_reviewsDF['pos_review_count'] = num_reviewsDF['pos_reviews'].apply(lambda x: findnum(str(x)))
    num_reviewsDF['neg_review_count'] = num_reviewsDF['neg_reviews'].apply(lambda x: findnum(str(x)))
    num_reviewsDF['total_review_count'] = num_reviewsDF['pos_review_count'] + num_reviewsDF['neg_review_count'] 

    final_data1 = final_data1.merge(num_reviewsDF, on = 'review_id', how='left')
    final_data1.to_csv(r"C:\\Users\\mabraham\\Documents\\IRI\\Sentiment\\Development\\sia_automation_am_revlink_scraping\\outputs\\sample_review_link.csv", index=False)
    return final_data1