"""
Created on Sat Dec 21 15:10:49 2019
@author: Yashwanth Ramachandra
"""

from facebook_scraper import get_posts
import logging
logger = logging.getLogger('sialogger')


"""
Facebook-Scrapper
"""
def fb_scraper(keyword,pages):
    data = []
    for page_i in range(1, int(pages)):
        try:
            if page_i % 1 == 0:
                logger.info("Pages %d of %d pages has been processed" % (page_i,int(pages)))
            for post in get_posts(keyword, pages=page_i, timeout=20):
                data.append(post)
        except AttributeError as e:
            logger.info("Error: {}".format(e))
            pass

    return data, keyword