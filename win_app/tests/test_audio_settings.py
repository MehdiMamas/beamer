import unittest
from dataclasses import replace

from app_config import ConfigError, config_from_dict, config_to_dict, default_config, validate_config


class AudioSettingsTests(unittest.TestCase):
    def test_defaults(self):
        config = default_config()
        self.assertTrue(config.audio_opus)
        self.assertTrue(config.audio_echo_cancel)
        self.assertEqual(config.audio_apps, [])
        self.assertFalse(config.audio_alerts)
        self.assertFalse(config.audio_sync_volume)

    def test_round_trip(self):
        config = replace(default_config(), audio_opus=False, audio_apps=["firefox", "spotify"],
                         audio_alerts=True, audio_sync_volume=True)
        again = config_from_dict(config_to_dict(config))
        validate_config(again)
        self.assertEqual(again.audio_apps, ["firefox", "spotify"])
        self.assertFalse(again.audio_opus)
        self.assertTrue(again.audio_alerts and again.audio_sync_volume)

    def test_bad_values_refused(self):
        base = default_config()
        for change in ({"audio_opus": "yes"}, {"audio_alerts": 1}, {"audio_apps": "firefox"},
                       {"audio_apps": ["x"] * 33}, {"audio_apps": [""]}, {"audio_apps": [5]},
                       {"audio_apps": ["a" * 65]}):
            with self.assertRaises(ConfigError, msg=str(change)):
                validate_config(replace(base, **change))


if __name__ == "__main__":
    unittest.main()
