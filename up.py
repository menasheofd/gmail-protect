import ast
import base64
import time

from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import re
import urllib.request
from datetime import datetime
from time import gmtime, strftime
import json
import whois
import xmltodict
import dns.resolver
from bs4 import BeautifulSoup
from urllib.parse import urlencode
from sklearn.metrics import accuracy_score
from scipy.io import arff
from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import itertools
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from pathlib import Path
import pickle
import requests
import json
import sqlite3

PHISHING_BASE_FILE = "TrainingDataset.arff"
PHISHING_NEW_FILE = "newTraningDateset.arff"
SPAM_FILE = "spamDataset.arff"
NOT_FOUND = -1
PHISHING = '-1'.encode('utf-8')
LEGITIMATE = '1'.encode('utf-8')
SUSPICIOUS = '0'.encode('utf-8')
PHISHING_RETURN = -1
NOT_FOUND_RETURN = 0
LEGITIMATE_RETURN = 1
TEN_HOURS = 32000


class BaseClassifier:
    clf = None

    def __init__(self, spam_or_phishing='phishing', data=None, label=None):
        self.data = []
        self.label = []
        if data is None or label is None:
            self.get_data(spam_or_phishing)
            # self.load_pickle(spam_or_phishing)
            self.train()
        else:
            self.data = data
            self.label = label
            self.train()

    def train(self):
        self.clf.fit(self.data, self.label)

    def predict(self, test_set):
        test_set = np.asarray(test_set, dtype=float)
        pred = self.clf.predict(test_set)
        proba = self.clf.predict_proba(test_set)
        return pred, proba

    def accuracy(self, pred):
        return accuracy_score(self.label, pred)

    def save_pickle(self, spam_or_phishing):
        with open(Path("pkl\\" + self.__class__.__name__ + "_" + spam_or_phishing + ".pkl"), 'wb') as output:
            pickle.dump(self.clf, output, pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, spam_or_phishing):
        if Path("pkl\\" + self.__class__.__name__ + "_" + spam_or_phishing + ".pkl").is_file():
            with open("pkl\\" + self.__class__.__name__ + "_" + spam_or_phishing + ".pkl", 'rb') as input:
                self.clf = pickle.load(input)
        else:
            self.train()
            self.save_pickle(spam_or_phishing)

    def get_data(self, spam_or_phishing):
        file = SPAM_FILE if spam_or_phishing == 'spam' else PHISHING_NEW_FILE
        all_data, meta = arff.loadarff(file)
        for row in all_data:
            self.data.append(list(row)[:-1])
            self.label.append(row[-1])
        self.data = np.asarray(self.data, dtype=float)
        self.label = np.asarray(self.label, dtype=float)

    def val_score(self):
        cv = ShuffleSplit(n_splits=1, test_size=0.2)
        scores = cross_val_score(self.clf, self.data, self.label, cv=cv)
        return sum(scores) / float(len(scores))

    @staticmethod
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """This function prints and plots the confusion matrix"""
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def confusion_matrix(self, test_label, prediction, class_names):
        """Compute confusion matrix"""
        cnf_matrix = metrics.confusion_matrix(test_label, prediction)
        np.set_printoptions(precision=2)  # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=class_names, title=self.__class__.__name__)
        # plt.show()
        # plt.savefig('confusion_matrix_' + self.__class__.__name__ + '.png')

    @staticmethod
    def roc(test_label, probas):
        """Compute ROC curve and area the curve"""
        fpr, tpr, thresholds = roc_curve(test_label, probas[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label='ROC fold(AUC = %0.2f)' % roc_auc, color='darkorange')
        lw = 2
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        # plt.savefig('ROC_' + self.__class__.__name__ + '.png')
        # plt.show()
        return roc_auc


class SvmClassifier(BaseClassifier):

    def __init__(self, spam_or_phishing='phishing', C=10.7, kernel='poly', degree=5, data=None, label=None):
        self.clf = svm.SVC(C=C, kernel=kernel, degree=degree, probability=True, gamma='scale')
        super().__init__(spam_or_phishing, data, label)


class PerceptronClassifier(BaseClassifier):

    def __init__(self, spam_or_phishing='phishing', penalty="l1", fit_intercept=False, max_iter=400, shuffle=False,
                 eta0=0.1,
                 data=None, label=None):
        self.clf = Perceptron(penalty=penalty, fit_intercept=fit_intercept, max_iter=max_iter, shuffle=shuffle,
                              eta0=eta0, random_state=0, tol=1e-3)
        super().__init__(spam_or_phishing, data, label)

    def predict(self, test_set):
        new_test_set = []
        for row in test_set:
            row = list(row)
            row.append('1'.encode('utf-8'))
            new_test_set.append(row)
        test_set = np.asarray(new_test_set, dtype=float)
        pred = self.clf.predict(test_set)
        return pred, None

    def get_data(self, spam_or_phishing):
        file = SPAM_FILE if spam_or_phishing == 'spam' else PHISHING_NEW_FILE
        all_data, meta = arff.loadarff(file)
        for row in all_data:
            data = (list(row)[:-1])
            data.append('1'.encode('utf-8'))
            self.data.append(data)
            self.label.append(row[-1])
        self.data = np.asarray(self.data, dtype=float)
        self.label = np.asarray(self.label, dtype=float)


class KNNClassifier(BaseClassifier):

    def __init__(self, spam_or_phishing='phishing', n_neighbors=8, weights='distance', algorithm='kd_tree', data=None,
                 label=None):
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
        super().__init__(spam_or_phishing, data, label)


class DecisionTreeClassifier(BaseClassifier):
    clf = tree.DecisionTreeClassifier()


class RForestClassifier(BaseClassifier):
    clf = RandomForestClassifier(n_estimators=100)


class GBoostingClassifier(BaseClassifier):
    clf = GradientBoostingClassifier()


class SpamFeatures:

    @staticmethod
    def get_features(data):
        features = []
        words_to_check = ["make", "address", "all", "3d", "our", "over", "remove", "internet", "order", "mail",
                          "receive", "will", "people", "report", "addresses", "free", "business", "email", "you",
                          "credit", "your", "font", "000", "money", "hp", "hpl", "george", "650", "lab", "labs",
                          "telnet", "857", "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                          "meeting", "original", "project", "re", "edu", "table", "conference"]
        chars_to_check = [';', '(', '[', '!', '$', '#']
        capital_letters_sequences = SpamFeatures.get_all_capital_letters_sequences_lengths(data)
        words_in_data = SpamFeatures.get_words(data)
        for word_to_check in words_to_check:
            num = SpamFeatures.get_word_freq(words_in_data, word_to_check)
            features.append(num)
        for char in chars_to_check:
            num = SpamFeatures.get_char_freq(data, char)
            features.append(num)
        features.append(SpamFeatures.avg_length(capital_letters_sequences))
        features.append(SpamFeatures.longest_sequence(capital_letters_sequences))
        features.append(SpamFeatures.get_num_of_capital_letters(data))
        return features

    @staticmethod
    def get_words(data):
        data = data.lower()
        return data.split()

    @staticmethod
    def get_word_freq(words, word):
        num = words.count(word)
        if len(words) == 0:
            return 0
        return 100 * (num / len(words))

    @staticmethod
    def get_char_freq(data, char):
        data = data.lower()
        num = data.count(char)
        if len(data) == 0:
            return 0
        return 100 * (num / len(data))

    @staticmethod
    def get_num_of_capital_letters(data):
        if len(data) == 0:
            return 0
        return sum(1 for letter in data if letter.isupper())

    @staticmethod
    def get_all_capital_letters_sequences_lengths(data):
        capital_letters_sequences_lengths = []
        length = 0
        flag = False

        for letter in data:
            if letter.isupper():
                length += 1
                flag = True

            else:
                if flag:
                    capital_letters_sequences_lengths.append(length)
                    length = 0
                flag = False

        if flag:
            capital_letters_sequences_lengths.append(length)

        return capital_letters_sequences_lengths

    @staticmethod
    def longest_sequence(capital_letters_sequences_lengths):
        capital_letters_sequences_lengths.sort(reverse=True)
        if len(capital_letters_sequences_lengths) == 0:
            return 0
        return capital_letters_sequences_lengths[0]

    @staticmethod
    def avg_length(capital_letters_sequences_lengths):
        if len(capital_letters_sequences_lengths) == 0:
            return 0
        return sum(capital_letters_sequences_lengths) / len(capital_letters_sequences_lengths)


regex = re.compile(
    r'^(?:http|ftp)s?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
    r'localhost|'  # localhost
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ip
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)


class PhishingFeatures:

    @staticmethod
    def get_features(url):
        return np.asarray([PhishingFeatures.have_ip_in_url(url), PhishingFeatures.url_length(url),
                           PhishingFeatures.shortining_service(url),
                           PhishingFeatures.having_strudel_sign(url),
                           PhishingFeatures.redirecting_using_double_slash(url),
                           PhishingFeatures.include_hyphen_symbol(url),
                           PhishingFeatures.sub_domain_and_multi_sub_domains(url),
                           PhishingFeatures.check_expiration_time(url),
                           PhishingFeatures.https_token_in_the_domain_part(url),
                           PhishingFeatures.request_url(url), PhishingFeatures.url_of_anchor(url),
                           PhishingFeatures.links_in_tags(url),
                           PhishingFeatures.age_of_domain(url), PhishingFeatures.dns_record(url),
                           PhishingFeatures.website_traffic(url),
                           PhishingFeatures.google_index(url)])

    @staticmethod
    def find_in_html(url, tag_name):
        html_text = urllib.request.urlopen(url)
        soup = BeautifulSoup(html_text, features="html.parser")
        tags = soup.find_all(tag_name)
        return tags

    @staticmethod
    def get_domain_part(url):
        split_url = url.split("://")
        i = (0, 1)[len(split_url) > 1]
        domain_part = split_url[i].split("?")[0].split('/')[0].split(':')[0].lower()
        return domain_part

    @staticmethod
    def have_ip_in_url(url):
        ip_in_url = re.findall(r'[0-9]+(?:\.[0-9]+){3}', url)
        hexadecimal_ip_in_url = re.findall(r'0x[0-9 A-F]+(?:\.0x[0-9 A-F]+){3}', url)
        return PHISHING if (len(ip_in_url) is not 0 or len(hexadecimal_ip_in_url) is not 0) else LEGITIMATE

    @staticmethod
    def url_length(url):
        length = len(url)
        if length < 54:
            return LEGITIMATE
        elif 54 <= length <= 75:
            return SUSPICIOUS
        else:
            return PHISHING

    @staticmethod
    def shortining_service(url):
        return PHISHING if (url.find("goo.gl") is not NOT_FOUND or url.find("bit.ly") is not NOT_FOUND
                            or url.find("tinyurl.com") is not NOT_FOUND or url.find("is.gd") is not NOT_FOUND
                            or url.find("dapalan.com") is not NOT_FOUND or url.find("bit.do") is not NOT_FOUND
                            or url.find("brand.link") is not NOT_FOUND or url.find("so.pr") is not NOT_FOUND
                            or url.find("ow.ly") is not NOT_FOUND or url.find("moourl.com") is not NOT_FOUND
                            or url.find("rebrand.ly") is not NOT_FOUND) else LEGITIMATE

    @staticmethod
    def having_strudel_sign(url):
        return PHISHING if url.find('@') is not NOT_FOUND else LEGITIMATE

    @staticmethod
    def redirecting_using_double_slash(url):
        return PHISHING if url.rfind("//") > 7 else LEGITIMATE

    @staticmethod
    def include_hyphen_symbol(url):
        domain_part = PhishingFeatures.get_domain_part(url)
        return PHISHING if domain_part.find('-') is not NOT_FOUND else LEGITIMATE

    @staticmethod
    def sub_domain_and_multi_sub_domains(url):
        domain_part = PhishingFeatures.get_domain_part(url)
        num_of_dots = domain_part.count('.')
        if url.find("www.") is NOT_FOUND:
            if num_of_dots <= 2:
                return LEGITIMATE
            elif num_of_dots is 3:
                return SUSPICIOUS
            else:
                return PHISHING
        else:
            if num_of_dots <= 3:
                return LEGITIMATE
            elif num_of_dots is 4:
                return SUSPICIOUS
            else:
                return PHISHING

    @staticmethod
    def check_expiration_time(url):
        domain_part = PhishingFeatures.get_domain_part(url)
        try:
            domain = whois.whois(domain_part)
            current_year = strftime("%Y-%m-%d", gmtime())
            expiration_date = str(domain['expiration_date']).split(' ')[0]
            expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
            current_year = datetime.strptime(current_year, "%Y-%m-%d")
            expiration_time = abs((expiration_date - current_year).days) / 365.0
            print(expiration_time)
            return PHISHING if expiration_time <= 1 else LEGITIMATE
        except:
            return PHISHING

    @staticmethod
    def https_token_in_the_domain_part(url):
        domain_part = PhishingFeatures.get_domain_part(url)
        return LEGITIMATE if domain_part.find("https") is NOT_FOUND else PHISHING

    @staticmethod
    def request_url(url):
        count = 0
        try:
            tags = PhishingFeatures.find_in_html(url, ["source", "img"])
        except:
            return PHISHING

        if len(tags) is 0:
            return LEGITIMATE

        for tag in tags:
            if tag.has_attr("src"):
                if re.match(regex, tag["src"]) is not None:
                    count += 1

        prec = count / len(tags) * 100

        if prec < 21:
            return LEGITIMATE

        elif 21 <= prec <= 61:
            return SUSPICIOUS

        return PHISHING

    @staticmethod
    def url_of_anchor(url):
        count = 0
        try:
            tags = PhishingFeatures.find_in_html(url, ["a"])
        except:
            return PHISHING
        if len(tags) == 0:
            return LEGITIMATE
        for tag in tags:
            if tag.has_attr("href"):
                href = tag["href"]
                if href == "#" or href == "#content" or href == "#skip" or href == "JavaScript ::void(0)":
                    count += 1

        prec = count / len(tags) * 100
        if prec < 31:
            return LEGITIMATE
        elif 31 <= prec <= 67:
            return SUSPICIOUS
        return PHISHING

    @staticmethod
    def links_in_tags(url):
        count1 = 0
        try:
            tags = PhishingFeatures.find_in_html(url, ["script", "meta", "link"])
        except:
            return PHISHING
        count2 = len(tags)
        if len(tags) == 0:
            return LEGITIMATE
        for tag in tags:
            if tag.name == "meta" and tag.has_attr("content"):
                content = tag["content"]
                if re.match(regex, content) is not None:
                    count1 += 1

            elif tag.name == "script":
                links = re.findall(regex, tag.text)
                count1 += len(links)

            elif tag.name == "link":
                if tag.has_attr("href") and tag["href"]:
                    count1 += 1

        prec = (count1 / count2) * 100

        if prec < 17:
            return LEGITIMATE

        elif 17 <= prec <= 81:
            return SUSPICIOUS

        return PHISHING

    @staticmethod
    def age_of_domain(url):
        domain_part = PhishingFeatures.get_domain_part(url)
        try:
            domain = whois.whois(domain_part)
            current_year = strftime("%Y-%m-%d", gmtime())
            creation_date = str(domain['creation_date']).split(' ')[0]
            creation_date = datetime.strptime(creation_date, "%Y-%m-%d")
            current_year = datetime.strptime(current_year, "%Y-%m-%d")
            age_of_domain = abs((current_year - creation_date).days) / 365.0
            return LEGITIMATE if age_of_domain >= 6 else PHISHING
        except:
            return PHISHING

    @staticmethod
    def dns_record(url):
        domain = PhishingFeatures.get_domain_part(url)
        try:
            answers = dns.resolver.query(domain, "NS")  # if its dont found  its throw exception
            return LEGITIMATE
        except:
            return PHISHING

    @staticmethod
    def website_traffic(url):
        try:
            domain_part = PhishingFeatures.get_domain_part(url)
            xml = urllib.request.urlopen('http://data.alexa.com/data?cli=10&dat=s&url={}'.format(domain_part)).read()
            xml_result = xmltodict.parse(xml)
            rank_data = json.dumps(xml_result).replace("@", "")
            data_to_json = json.loads(rank_data)
            site_rank = int(data_to_json["ALEXA"]["SD"][1]["POPULARITY"]["TEXT"])
            if site_rank is 0:
                return PHISHING
            elif site_rank < 100000:
                return LEGITIMATE
            else:
                return SUSPICIOUS
        except:
            return PHISHING

    @staticmethod
    def google_index(url):
        user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116' \
                     ' Safari/537.36 '
        headers = {'User-Agent': user_agent}
        query = {'q': 'info:' + url}
        google = "https://www.google.com/search?" + urlencode(query)
        try:
            data = requests.get(google, headers=headers)
        except:
            return PHISHING
        data.encoding = 'ISO-8859-1'
        soup = BeautifulSoup(str(data.content), "html.parser")
        try:
            check = soup.find(id="rso").find("div").find("div").find("div").find("div").find("a")
            href = check['href']  # if its not exist throw exception
            return LEGITIMATE
        except AttributeError:
            return PHISHING


class SqliteDB:
    def __init__(self):
        self.con = sqlite3.connect(":memory:")
        self.db = self.con.cursor()
        self.db.execute(
            'CREATE TABLE IF NOT EXISTS list (ID INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,gmail_address TEXT, '
            'url TEXT,  legitimate int)')

    def delete_url_from_list(self, url):
        self.db.execute("DELETE FROM list where url == " + url)
        self.con.commit()

    def insert_data_to_list(self, gmail_address, url, ml_answer):
        """insert the data to the DB black list table"""
        self.db.execute("insert into list (gmail_address, url, legitimate) values (?, ?, ?)",
                       (gmail_address, url, ml_answer))
        self.con.commit()

    def gmail_address_in_list(self, gmail_address):
        """check if the gmail address in the list table"""
        phishing_count = len(list(self.db.execute("SELECT gmail_address FROM list where gmail_address =? and "
                                                 "legitimate = -1", (gmail_address,))))

        return PHISHING_RETURN if phishing_count >= 20 else NOT_FOUND_RETURN

    def search_urls_in_db(self, urls):
        for url in urls:
            db_answer = self.url_in_list(url)
            if db_answer is not NOT_FOUND_RETURN:
                return db_answer
        return NOT_FOUND_RETURN

    def url_in_list(self, url):
        """check if the url in the list table"""
        legitimate_count = 0
        if url is None:
            return NOT_FOUND_RETURN
        phishing_count = len(list(self.db.execute("SELECT url FROM list where url =?  and legitimate = -1", (url,))))
        if phishing_count == 0:
            legitimate_count = len(list(self.db.execute("SELECT url FROM list where url =?  and legitimate = 1", (url,))))
        elif phishing_count is not 0:
            return PHISHING_RETURN
        return LEGITIMATE_RETURN if legitimate_count is not 0 else NOT_FOUND_RETURN


class Manager:

    def __init__(self):
        self.svm = SvmClassifier('phishing')
        self.perceptron = PerceptronClassifier('phishing')
        self.knn = KNNClassifier('phishing')
        self.decision_tree = DecisionTreeClassifier('phishing')
        self.random_forest_spam = RForestClassifier('spam')
        self.db = SqliteDB()

    @staticmethod
    def get_spam_features_str(features, str_result='1'):
        features_str = str(features)
        features_str = features_str.replace('[', '')
        features_str = features_str.replace(']', '')
        features_str = features_str.replace('b', '')
        features_str = features_str.replace("'", '')
        features_str = features_str.replace('  ', ',')
        features_str = features_str.replace(' ', '')
        features_str = features_str.replace('\n', '')
        features_str = features_str + "," + str_result + '\n'
        return features_str

    @staticmethod
    def get_phishing_features_str(features, str_result='1'):
        features_str = str(features)
        features_str = features_str.replace('[', '')
        features_str = features_str.replace(']', '')
        features_str = features_str.replace('b', '')
        features_str = features_str.replace("'", '')
        features_str = features_str.replace(' ', ',')
        features_str = features_str.replace('\n', '')
        features_str = features_str + "," + str_result + '\n'
        return features_str

    @staticmethod
    def replace_line_from_file(features_str, spam_or_phishing):
        file = SPAM_FILE if spam_or_phishing == 'spam' else PHISHING_NEW_FILE
        new_features_str = features_str[:-2] + '1'
        f = open(file, 'r+')
        data = f.readlines()
        f.seek(0)
        for line in data:
            line = line.replace(features_str, new_features_str)
            f.write(line)
        f.truncate()
        f.close()

    @staticmethod
    def add_to_file(features_str, spam_or_phishing):
        file = SPAM_FILE if spam_or_phishing == 'spam' else PHISHING_NEW_FILE
        f = open(file, 'a')
        f.write(features_str)
        f.close()

    def check_spam(self, data):
        # print(data)
        features = SpamFeatures.get_features(data)
        pred, proba = self.random_forest_spam.predict([features])
        Manager.add_to_file(Manager.get_spam_features_str(features, str(int(pred[0]))), 'spam')
        return int(pred[0])

    def check_phishing(self, gmail_address, urls):
        try:
            phishing = False
            # db_answer = self.check_in_db(urls, gmail_address)
            # if db_answer is not NOT_FOUND_RETURN:
            #     return db_answer

            for url in urls:
                if self.check_phishing_in_algorithms(url) == PHISHING_RETURN:
                    if gmail_address is not 'None':
                        pass
                        # self.db.insert_data_to_list(gmail_address, url, PHISHING_RETURN)
                    Manager.add_to_file(
                        self.get_phishing_features_str(PhishingFeatures.get_features(url), str(PHISHING_RETURN)),
                        'phishing')
                    phishing = True
                else:
                    if gmail_address is not 'None':
                        pass
                        # self.db.insert_data_to_list(gmail_address, url, LEGITIMATE_RETURN)
                    Manager.add_to_file(
                        self.get_phishing_features_str(PhishingFeatures.get_features(url), str(LEGITIMATE_RETURN)),
                        'phishing')
            return PHISHING_RETURN if phishing else LEGITIMATE_RETURN
        except Exception as e:
            return str(e)

    def check_in_db(self, urls, gmail_address):
        if self.db.gmail_address_in_list(gmail_address) is PHISHING_RETURN or self.db.search_urls_in_db(
                urls) is PHISHING_RETURN:
            return PHISHING_RETURN
        if self.db.search_urls_in_db(urls) is LEGITIMATE_RETURN:
            return LEGITIMATE_RETURN
        return NOT_FOUND_RETURN

    def check_phishing_in_algorithms(self, url):
        url_feature = [PhishingFeatures.get_features(url)]
        pred, proba = self.svm.predict(url_feature)
        percentage = proba[0][1] if pred[0] == LEGITIMATE_RETURN else proba[0][0]
        if percentage < 0.6 or pred[0] == PHISHING_RETURN:
            decision_tree_pred, decision_tree_proba = self.decision_tree.predict(url_feature)
            knn_pred, knn_proba = self.knn.predict(url_feature)
            perceptron_pred, perceptron_proba = self.perceptron.predict(url_feature)
            result = decision_tree_pred[0] + knn_pred[0] + perceptron_pred[0]
            if pred[0] == LEGITIMATE_RETURN:
                if result >= 1:  # doing best of 3
                    return LEGITIMATE_RETURN
                else:
                    return PHISHING_RETURN  # if self.check_API(url) is True else LEGITIMATE_RETURN

            if pred[0] == PHISHING_RETURN:
                if percentage < 0.6:
                    if result <= -1:  # doing best of 3
                        return PHISHING_RETURN  # if self.check_API(url) is True else LEGITIMATE_RETURN
                    else:
                        return LEGITIMATE_RETURN
                else:
                    if result <= 1:  # if one of them choose
                        return PHISHING_RETURN  # if self.check_API(url) is True else LEGITIMATE_RETURN
                    else:
                        return LEGITIMATE_RETURN
        else:
            return int(pred[0])


manger = Manager()
app = Flask(__name__)
stop = False


def encrypt(msg_to_sent):
    """encrypted the msg with base64 method"""
    return str(base64.b64encode(msg_to_sent.encode('ascii')))[2:-1]


def decrypt(encrypted_msg):
    """decrypted the msg with base64 method"""

    return str(base64.b64decode(encrypted_msg))[2:-1]


@app.route('/spam', methods=['GET', 'DELETE'])
def spam():
    while stop:
        pass
    data = request.get_json()
    mail_data = data['mail_data']
    if request.method == 'GET':
        return str(manger.check_spam(mail_data))
    else:
        manger.replace_line_from_file(manger.get_spam_features_str(SpamFeatures.get_features(mail_data), '-1'), 'spam')


@app.route('/')
def start_msg():
    return "check phishing and spam api!!!!!"


@app.route('/spam_and_phishing', methods=['GET'])
def spam_and_phishing():
    while stop:
        pass
    data = request.get_json()
    mail_data = data['mail_data']
    gmail_address = data['gmail_address']
    urls = data['urls']
    print(urls)
    return jsonify({'phishing': manger.check_phishing(gmail_address, urls), 'spam': manger.check_spam(mail_data)})


@app.route('/phishing', methods=['GET', 'DELETE'])
def phishing():
    while stop:
        pass
    data = request.get_json()
    url = data['url']
    if request.method == 'GET':
        return str(manger.check_phishing('None', [url]))
    else:
        manger.replace_line_from_file(manger.get_phishing_features_str(PhishingFeatures.get_features(url), '-1'),
                                      'phishing')
        manger.db.delete_url_from_list(url)


def update_ml_algoritems():
    global stop
    while True:
        time.sleep(TEN_HOURS)
        stop = True
        time.sleep(10)  # in seconds
        manger.svm = SvmClassifier('phishing')
        manger.perceptron = PerceptronClassifier('phishing')
        manger.knn = KNNClassifier('phishing')
        manger.decision_tree = DecisionTreeClassifier('phishing')
        manger.random_forest_spam = RForestClassifier('spam')
        stop = False


if __name__ == '__main__':
    app.run()
