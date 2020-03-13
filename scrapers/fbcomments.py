"""
Created on Wed Jan  8 10:25:55 2020
@author: AkOjha
"""

import pandas as pd


def scrapeFbComments(post, driver):
          
    driver.get(post)
    ids = driver.find_elements_by_xpath("//div[@class='_2a_i']")
    number_of_comments = len(ids)
    comment = []
    authorname = []
    for i in range(0,number_of_comments):
        
        authorname.append(driver.find_elements_by_xpath("//div[@class='_2b05']")[i].text)
        comment.append(driver.find_elements_by_xpath("//div[@class='_2a_i']")[i].text)
              
    final_data = pd.DataFrame({'authorName':authorname, 'commentWithAuthorname':comment, 'post':post})
    return final_data