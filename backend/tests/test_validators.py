"""The deterministic checks. These are the parts that must never silently regress."""
import pytest

from verityne.detectors.id_forensics import validate_pan
from verityne.utils.ocr import find_fields, normalise_pan
from verityne.utils.verhoeff import append_verhoeff, validate_aadhaar


class TestVerhoeff:
    def test_generated_numbers_validate(self):
        for body in ["23456789012", "98765432101", "45612378909"]:
            assert validate_aadhaar(append_verhoeff(body))

    def test_wrong_check_digit_fails(self):
        good = append_verhoeff("23456789012")
        bad = good[:-1] + str((int(good[-1]) + 1) % 10)
        assert not validate_aadhaar(bad)

    def test_rejects_leading_zero_or_one(self):
        # UIDAI never issues a number starting 0 or 1.
        assert not validate_aadhaar(append_verhoeff("01234567890"))
        assert not validate_aadhaar(append_verhoeff("11234567890"))

    def test_rejects_wrong_length(self):
        assert not validate_aadhaar("2345678901")


class TestPan:
    def test_valid_pan_passes(self):
        r = validate_pan("ABCPK1234X", claimed_name="Ravi Kumar")
        assert r["format_ok"] and r["holder_type_ok"] and r["surname_initial_ok"]
        assert not r["issues"]

    def test_invalid_holder_type_is_caught(self):
        # 'X' is not a valid 4th-character holder code.
        r = validate_pan("ABCXK1234X", claimed_name="Ravi Kumar")
        assert r["format_ok"] and not r["holder_type_ok"]
        assert any("holder-type" in i for i in r["issues"])

    def test_surname_initial_mismatch_is_caught(self):
        r = validate_pan("ABCPZ1234X", claimed_name="Ravi Kumar")
        assert r["surname_initial_ok"] is False
        assert any("surname initial" in i for i in r["issues"])

    def test_malformed_pan_rejected(self):
        assert not validate_pan("AB1PK234X")["format_ok"]

    def test_surname_check_skipped_without_a_name(self):
        r = validate_pan("ABCPK1234X")
        assert r["surname_initial_ok"] is None


class TestOcrRepair:
    def test_positional_confusion_repair(self):
        # easyocr routinely reads 0 as O inside the numeric block.
        assert normalise_pan("QYSPP17O8U") == "QYSPP1708U"

    def test_finds_repaired_pan_in_noisy_text(self):
        text = "INCOME TAX DEPARTMENT Permanent Account Number QYSPP17O8U Signature"
        assert find_fields(text)["pan"] == "QYSPP1708U"

    def test_no_false_pan_from_ordinary_text(self):
        assert find_fields("GOVERNMENT OF INDIA DEPARTMENT")["pan"] is None

    def test_extracts_dob(self):
        assert find_fields("Date of Birth 19/02/1974")["dob"] == "19/02/1974"
