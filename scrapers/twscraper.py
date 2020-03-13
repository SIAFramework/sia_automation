"""
Created on Sat Dec 21 15:10:49 2019
@author: Yashwanth Ramachandra
"""

from twitter_scraper import get_tweets
import logging
logger = logging.getLogger('sialogger')


"""
Twitter-Scrapper
"""
def tw_scraper(keyword,pages):
    data = []
    page_i=int(pages)
    try:
        if page_i % 1 == 0:
            logger.info("Pages %d of %d has been processed" % (page_i,int(pages)))
        for post in get_tweets(keyword, pages=page_i):
            data.append(post)
    except AttributeError as e:
        logger.info("Error: {}".format(e))
        pass
    return data, keyword