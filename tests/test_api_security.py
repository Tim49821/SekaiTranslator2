import unittest

from utils.api_security import is_loopback_host, require_auth_for_public_bind


class ApiSecurityTest(unittest.TestCase):
    def test_loopback_hosts_are_recognized(self):
        for host in ('localhost', '127.0.0.1', '127.0.0.42', '::1', '[::1]'):
            with self.subTest(host=host):
                self.assertTrue(is_loopback_host(host))

    def test_public_hosts_require_every_declared_token(self):
        for host in ('0.0.0.0', '::', 'api.example.com', ''):
            with self.subTest(host=host):
                with self.assertRaisesRegex(ValueError, 'Missing token.*worker'):
                    require_auth_for_public_bind(host, {'client': 'client-token', 'worker': ''})

    def test_public_host_accepts_configured_tokens(self):
        require_auth_for_public_bind(
            '0.0.0.0',
            {'client': 'client-token', 'worker': 'worker-token'},
        )

    def test_explicit_unsafe_override_is_required_for_empty_tokens(self):
        require_auth_for_public_bind(
            '0.0.0.0',
            {'client': '', 'worker': ''},
            allow_unauthenticated_public=True,
        )


if __name__ == '__main__':
    unittest.main()
