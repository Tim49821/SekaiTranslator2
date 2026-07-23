from typing import Dict, List

import requests

from .base import PROXY, BaseTranslator, register_translator


PAPAGO_TRANSLATE_URL = 'https://papago.naver.com/api/text/translation'
PAPAGO_REQUEST_TIMEOUT = 15


@register_translator('Papago')
class PapagoTranslator(BaseTranslator):

    concate_text = True
    params: Dict = {'delay': 0.0}

    def _setup_translator(self):
        # Papago's current web API no longer uses the PPG key embedded in a
        # main.js bundle. Keep setup local so selecting the translator does not
        # depend on scraping Papago's changing frontend asset names.
        self.lang_map['Auto'] = 'auto'
        self.lang_map['简体中文'] = 'zh-CN'
        self.lang_map['繁體中文'] = 'zh-TW'
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'
        self.lang_map['한국어'] = 'ko'
        self.lang_map['Tiếng Việt'] = 'vi'
        self.lang_map['Français'] = 'fr'
        self.lang_map['Deutsch'] = 'de'
        self.lang_map['Italiano'] = 'it'
        self.lang_map['Português'] = 'pt'
        self.lang_map['русский язык'] = 'ru'
        self.lang_map['Español'] = 'es'
        self.lang_map['Thai'] = 'th'
        self.lang_map['Arabic'] = 'ar'
        self.lang_map['Hindi'] = 'hi'

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []

        data = {
            'source': self.lang_map[self.lang_source],
            'target': self.lang_map[self.lang_target],
            'text': src_list[0],
            'dict': 'false',
            'useGlossary': 'false',
            'honorific': 'false',
        }
        headers = {
            'Accept-Language': 'ko',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        }

        response = requests.post(
            PAPAGO_TRANSLATE_URL,
            data=data,
            headers=headers,
            proxies=PROXY,
            timeout=PAPAGO_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        response_data = response.json()
        translated_text = (
            response_data.get('translatedText')
            if isinstance(response_data, dict)
            else None
        )
        if not isinstance(translated_text, str):
            error_code = (
                response_data.get('errorCode')
                if isinstance(response_data, dict)
                else None
            )
            detail = f' (error code: {error_code})' if error_code else ''
            raise RuntimeError(f'Papago returned an invalid translation response{detail}.')

        return [translated_text]
