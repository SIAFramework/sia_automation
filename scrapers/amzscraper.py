"""
Created on Tue Nov 12 11:06:44 2019
@author: Amit Ojha
"""

import pandas as pd
import logging
logger = logging.getLogger('sialogger')


"""
Create the Linkset
"""
def create_linkset(row):
    
    num_rev = int(row.Review_Count)
    link = str(row.Review_Link_Href)
    link2 = []
    pages = int(num_rev/10)+1
    cutoff = link.find("show_all_btm?ie=UTF8&reviewerType=all_reviews")
    link1 = link[:cutoff]
    for i in range(1,pages+1):
        temp = 'paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber='+str(i)
        link2.append(temp)
    linkset = [link1+x for x in link2]
    return linkset

 
"""
Scrape Reviews
"""
def scrap_reviews(link, driver):

    driver.get(link)
    ids = driver.find_elements_by_xpath("//*[contains(@id,'customer_review-')]")
    comment_ids = []
    for i in ids:
        comment_ids.append(i.get_attribute('id'))

    authorname = []
    title = []
    body = []
    date = []
    usefullness = []

    for x in comment_ids:
        try:
            authorname_element = driver.find_elements_by_xpath('//*[@id="' + x +'"]/div[1]/a/div[2]/span')[0]
            authornametemp = authorname_element.text
            authorname.append(authornametemp)
        except Exception as e:
            logger.info("Exception is {}".format(e))
            authorname.append(None)

        try:
            title_element = driver.find_elements_by_xpath('//*[@id="' + x +'"]/div[2]/a[2]/span')[0]
            titletemp = title_element.text
            title.append(titletemp)
        except Exception as e:
            logger.info("Exception is {}".format(e))
            title.append(None)

        try:
            date_element = driver.find_elements_by_xpath('//*[@id="' + x +'"]/span')[0]
            datetemp = date_element.text
            date.append(datetemp)
        except Exception as e:
            logger.info("Exception is {}".format(e))
            date.append(None)

        try:
            usefullness_element = driver.find_elements_by_xpath('//*[@id="' + x +'"]/div[5]/div/span[1]/div[1]/span')[0]
            usefullnesstemp = usefullness_element.text
            usefullness.append(usefullnesstemp)
        except Exception as e:
            logger.info("Exception is {}".format(e))
            usefullness.append(None)

        try:
            body_element = driver.find_elements_by_xpath('//*[@id="' + x +'"]/div[4]/span/span')[0]
            bodytemp = body_element.text
            body.append(bodytemp)
        except Exception as e:
            logger.info("Exception is {}".format(e))
            body.append(None)    
    final_data = pd.DataFrame({'review_link':link, 'author':authorname, 'title':title, 'date':date, 'usefullness':usefullness, 'body':body})
    return final_data