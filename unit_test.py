import random
import unittest
from unittest.mock import Mock, patch
import my_class
import requests
from bs4 import BeautifulSoup as bs


#my_class.Scraper.get_soup(1)




class TestScraper(unittest.TestCase):

    #test for constructor
    def test_init_valid_input(self):
        scraper = my_class.Scraper(4, "palermo")
        self.assertIsInstance(scraper.pages, int)
        self.assertIsInstance(scraper.cities, str)

    def test_init_value_error_page(self):
        for _ in range(100):
            invalid_page = random.choice([random.randint(-100, 0),
                                          "".join(random.choices("abc def ghi", k=5))])
            with self.subTest(i=invalid_page):
             with self.assertRaises(ValueError):
                my_class.Scraper(invalid_page, "palermo")

    #+
    def test_init_page_default_value(self):
        scraper = my_class.Scraper(city = "palermo")
        self.assertEqual(scraper.pages,1)

    def test_init_value_error_city(self):
        with self.assertRaises(ValueError):
            my_class.Scraper(1,1)

    def test_city_missing(self):
        with self.assertRaises(ValueError):
            my_class.Scraper(1)

    def test_init_city_with_spaces(self):
        scraper = my_class.Scraper(4, "palermo centro")
        self.assertEqual(scraper.cities, "palermo-centro")




    #test for get_soup
    def test_input_get_soup_value_error_str(self):
        with self.assertRaises(ValueError):
            my_class.Scraper.get_soup(1)

    def test_input_get_soup_value_error_http(self):
        with self.assertRaises(ValueError):
            my_class.Scraper.get_soup("test")

    @patch("requests.get")
    @patch("my_class.bs")
    def test_output_get_soup(self, mock_bs, mock_get):
        mock_response = Mock()
        mock_response.text = "<html><body><h1>Test</h1></body></html>"
        mock_get.return_value = mock_response

        mock_bs.return_value = "mimmo"

        url = "http://example.com"
        test = my_class.Scraper.get_soup(url)

        self.assertEqual(test,"mimmo")


    @patch("my_class.Scraper.get_soup")
    #test for max_page
    def test_max_page_output(self,mock_get_soup):

        mock_soup = Mock()

        mock_pagination = Mock()

        disabled_item = Mock()

        disabled_item.get_text.return_value = "7"

        mock_pagination.find_all.return_value = [disabled_item]

        mock_soup.find.return_value = mock_pagination


        mock_get_soup.return_value = mock_soup


        scraper = my_class.Scraper(page=1, city="milano")

        result = scraper.max_page()
        self.assertEqual(result, 7)



















