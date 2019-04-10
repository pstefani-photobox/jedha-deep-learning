import flickrapi
import random, os, requests, json, time

class FlickrCrawler(object):
    """
    Crawls through Flickr and download images respecting pyTorch folder structures:
        concept1
            train
                image1
                image2
            test
        concept2
            train
            test
    """
    def __init__(self, apikey, apisecret, img_per_class=1000):
        self.apikey = apikey
        self.flickr = flickrapi.FlickrAPI(apikey, apisecret, format='parsed-json')
        self.img_per_class = img_per_class

    def query(self, entry):
        try:
            results = self.flickr.photos.search(text=entry,
                                            license='1',
                                            privacy_filter=1,
                                            sort='relevance')
        except:
            time.sleep(10)
            results = self.flickr.photos.search(text=entry,
                                                license='1',
                                                privacy_filter=1,
                                                sort='relevance')
        results = results.get('photos').get('photo')[:self.img_per_class]
        return results

    def parse_results(self, results):
        return [ 'https://farm{}.staticflickr.com/{}/{}_{}_z.jpg'.format(
                                                                  r.get('farm'),
                                                                  r.get('server'),
                                                                  r.get('id'),
                                                                  r.get('secret'))
            for r in results]

    def download_results(self, results, path, concept, train_test_percentage=90.0):
        random.shuffle(results)
        nb_train_images = int(train_test_percentage*len(results)/100)
        train_images = results[:nb_train_images]
        test_images = results[nb_train_images:]
        self.index = 0
        train_path = os.path.join(path, 'train')
        test_path = os.path.join(path, 'test')
        if not os.path.exists(path):
            raise OSError('No such path as {}'.format(path))
        else:

            if not os.path.exists(train_path):
                os.mkdir(train_path)
            if not os.path.exists(test_path):
                os.mkdir(test_path)

            train_concept_path = os.path.join(train_path, concept)
            if not os.path.exists(train_concept_path):
                os.mkdir(train_concept_path)
            test_concept_path = os.path.join(test_path, concept)
            if not os.path.exists(test_concept_path):
                os.mkdir(test_concept_path)
        self._download(train_images, train_concept_path)
        self._download(test_images, test_concept_path)

    def _download(self, urls, path):
        for url in urls:
            img = requests.get(url).content
            with open(os.path.join(path, 'img000{}.jpg'.format(self.index)),'wb') as file:
                file.write(img)
            self.index+=1

    def run(self, path, query):
        results = self.parse_results(self.query(query))
        self.download_results(results, path, query)


if __name__ == "__main__":

    config = json.load(open('crawler_config.json'))
    access_key, secret_key = config.get('api_key'), config.get('secret_key')
    crawler = FlickrCrawler(access_key.encode(), secret_key.encode(), config.get('img_per_class'))
    queries = config.get('queries')
    for q in queries:
        crawler.run(config.get('download_path'), q)
