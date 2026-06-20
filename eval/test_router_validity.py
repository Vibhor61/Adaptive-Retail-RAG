import unittest
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contracts.router_contracts import ValidityStatus, ValidityFlags
from routing_layer.validity import validate_query_structure, MAX_QUERY_LENGTH, HARD_REJECT_LENGTH

class TestQueryStructureValidation(unittest.TestCase):

    def test_01_clean_executable_query(self):
        """Should successfully parse a clean, standard user intent query."""
        query = "What is the battery capacity of the iDROID King Battery?"
        result = validate_query_structure(query)

        self.assertEqual(result.status, ValidityStatus.EXECUTABLE)
        self.assertEqual(len(result.anomaly_flags), 0)
        self.assertEqual(result.word_count, 10)

    def test_02_heavy_amazon_title_executable(self):
        """Should accept highly complex Amazon titles if they stay within normal boundaries."""
        query = (
            "Specs for Apple iPhone 15 Pro Max Clear Case with MagSafe wireless charging "
            "compatibility transparent rigid PC back 2024 model"
        )
        result = validate_query_structure(query)

        self.assertEqual(result.status, ValidityStatus.EXECUTABLE)
        self.assertEqual(len(result.anomaly_flags), 0)
        self.assertLess(result.word_count, 30)

    def test_03_empty_query_degraded(self):
        """Should return DEGRADED with EMPTY_QUERY flag for empty or whitespace-only inputs."""
        query = "     \n   \t  "
        result = validate_query_structure(query)

        self.assertEqual(result.status, ValidityStatus.DEGRADED)
        self.assertIn(ValidityFlags.EMPTY_QUERY, result.anomaly_flags)
        self.assertEqual(result.normalized_query, "")
        self.assertEqual(result.word_count, 0)

    def test_04_hard_length_rejection(self):
        """Should forcefully return DEGRADED if the query passes the extreme hard length ceiling."""
        query = "A " * (HARD_REJECT_LENGTH + 10)
        result = validate_query_structure(query)

        self.assertEqual(result.status, ValidityStatus.DEGRADED)
        self.assertIn(ValidityFlags.HARD_LENGTH_REJECT, result.anomaly_flags)
        self.assertEqual(len(result.normalized_query), 100)  # Truncated window check

    def test_05_control_characters_degraded(self):
        """Should gracefully degrade queries containing illegal, hidden control characters."""
        query = "What is the wattage \x00 of this hidden hack?"
        result = validate_query_structure(query)

        self.assertEqual(result.status, ValidityStatus.DEGRADED)
        self.assertIn(ValidityFlags.CONTROL_CHARACTERS, result.anomaly_flags)

    def test_06_excessive_length_suspicious(self):
        """Should flag query as SUSPICIOUS when length crosses soft MAX_QUERY_LENGTH limit."""
        base_query = "What features does this phone cover provide? "
        query = base_query * int((MAX_QUERY_LENGTH / len(base_query)) + 2)
        result = validate_query_structure(query)

        self.assertEqual(result.status, ValidityStatus.SUSPICIOUS)
        self.assertIn(ValidityFlags.EXCESSIVE_LENGTH, result.anomaly_flags)

    def test_07_repeated_symbol_spam(self):
        """Should flag consecutive non-alphanumeric token floods (e.g. !!!!)."""
        query = "Is this protective case shockproof???? I need to know now."
        result = validate_query_structure(query)

        self.assertEqual(result.status, ValidityStatus.SUSPICIOUS)
        self.assertIn(ValidityFlags.SYMBOL_SPAM, result.anomaly_flags)

    def test_08_character_flood_spam(self):
        """Should catch keyboard mashing or character dragging patterns (e.g., matching 7+ identical chars)."""
        query = "the charging speed on this dock is completely brokenhhhhhhh"
        result = validate_query_structure(query)

        self.assertEqual(result.status, ValidityStatus.SUSPICIOUS)
        self.assertIn(ValidityFlags.CHARACTER_FLOOD, result.anomaly_flags)

    def test_09_high_symbol_ratio(self):
        """Should catch code injection or heavy ascii noise templates by computing symbol percentage."""
        query = "@@##$$%%^^**__++==%%"
        result = validate_query_structure(query)

        self.assertEqual(result.status, ValidityStatus.SUSPICIOUS)
        self.assertIn(ValidityFlags.HIGH_SYMBOL_RATIO, result.anomaly_flags)

    def test_10_unicode_normalization(self):
        """Should perfectly flatten exotic Unicode font variants (NFKC) back to clean structural strings."""
        # Using bold/italic mathematical script characters for 'iPhone'
        exotic_query = "What is the price of ɩ𝖯𝗁𝗈𝗇𝖾?"
        result = validate_query_structure(exotic_query)

        self.assertEqual(result.status, ValidityStatus.EXECUTABLE)
        # Checking if '𝖯𝗁𝗈𝗇𝖾' normalizes down cleanly to alphanumeric standard space
        self.assertTrue(result.normalized_query.isprintable())


if __name__ == "__main__":
    unittest.main()