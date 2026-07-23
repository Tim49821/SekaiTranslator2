import unittest
from unittest.mock import Mock, patch

from modules.translators.trans_papago import (
    PAPAGO_REQUEST_TIMEOUT,
    PAPAGO_TRANSLATE_URL,
    PapagoTranslator,
)


class PapagoTranslatorTest(unittest.TestCase):
    def test_setup_does_not_scrape_frontend_assets(self):
        with patch('modules.translators.trans_papago.requests.get') as get:
            translator = PapagoTranslator('日本語', '한국어')

        get.assert_not_called()
        self.assertEqual(translator.lang_map['Auto'], 'auto')
        self.assertEqual(translator.lang_map['日本語'], 'ja')
        self.assertEqual(translator.lang_map['한국어'], 'ko')

    @patch('modules.translators.trans_papago.requests.post')
    def test_translate_uses_current_web_api(self, post):
        response = Mock()
        response.json.return_value = {'translatedText': '안녕하세요'}
        post.return_value = response
        translator = PapagoTranslator('日本語', '한국어')

        result = translator.translate('こんにちは')

        self.assertEqual(result, '안녕하세요')
        post.assert_called_once()
        _, kwargs = post.call_args
        self.assertEqual(post.call_args.args[0], PAPAGO_TRANSLATE_URL)
        self.assertEqual(
            kwargs['data'],
            {
                'source': 'ja',
                'target': 'ko',
                'text': 'こんにちは',
                'dict': 'false',
                'useGlossary': 'false',
                'honorific': 'false',
            },
        )
        self.assertEqual(kwargs['timeout'], PAPAGO_REQUEST_TIMEOUT)
        response.raise_for_status.assert_called_once_with()

    @patch('modules.translators.trans_papago.requests.post')
    def test_translate_rejects_api_error_response(self, post):
        response = Mock()
        response.json.return_value = {'errorCode': 'TEST-ERROR'}
        post.return_value = response
        translator = PapagoTranslator('日本語', '한국어')

        with self.assertRaisesRegex(RuntimeError, 'TEST-ERROR'):
            translator.translate('こんにちは')


if __name__ == '__main__':
    unittest.main()
