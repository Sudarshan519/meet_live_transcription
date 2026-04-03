from math import log
import requests
import json
import time
import hashlib
import hmac
import logging 
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load server API key from environment
SERVER_API_KEY = os.environ.get("SERVER_API_KEY")
HMAC_SECRET = os.environ.get("HMAC_SECRET")

class TokenReservationClient:
    """
    Client for interacting with the Reservation System API for token management.
    Updated to match working test patterns.
    """

    # Platform ID
    TOOL_CODE = "P1022"

    # Reservation API base URL (should be set in Django settings)
    BASE_URL =    "https://api.testir.xyz/portaltest/api"
   

    # Error message constants
    ERR_CHECK_AVAILABILITY = "Unable to check token availability"
    ERR_UNKNOWN = "Unknown error"
    ERR_INSUFFICIENT_TOKENS = "Insufficient tokens"
    ERR_FREE_TOOL = "Free tool - no tokens deducted"
    ERR_RESERVATION_FAILED = "Failed to reserve tokens"
    ERR_CONSUME_FAILED = "Failed to consume reservation"
    ERR_RELEASE_FAILED = "Failed to release reservation"
    MSG_TOKENS_RESERVED = "Tokens reserved successfully"
    MSG_TOKENS_CONSUMED = "Tokens consumed successfully"
    MSG_TOKENS_RELEASED = "Tokens released successfully"

    logger = logging.getLogger(__name__)

    @classmethod
    def _log_api_response_to_db(cls, **kwargs):
        """
        Stub for database logging. Course model references removed.
        """
        pass

    @classmethod
    def _create_signature(cls, payload):
        """
        Create HMAC SHA256 signature for the given payload.
        Matches the working test script pattern exactly.
        """
        cls.logger.info(f"Creating HMAC signature for payload: {payload}")
        payload_string = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        cls.logger.info(f"Payload string for signature: {payload_string}")
        signature = hmac.new(
            HMAC_SECRET.encode("utf-8"),
            payload_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        cls.logger.info(f"Generated signature: {signature}")
        return signature

    @classmethod
    def _headers(cls):
        headers = {
            "x-api-key": SERVER_API_KEY,
            "Content-Type": "application/json",
        }
        cls.logger.info(f"Generated headers: {headers}")
        return headers

    @classmethod
    def check_availability(
        cls,
        user_id,
        feature_id=None,
        metadata=None,
        custom_deduction=None,
        task_id=None,
        email=None,
        platform_id=None,
    ):
        """
        Check if user has enough tokens for the operation (reservation-aware).
        Updated to match working test script pattern.
        """
        timestamp = int(time.time() * 1000)
        used_platform_id = platform_id if platform_id else cls.TOOL_CODE
        payload = {
            "userId": str(user_id),
            "featureId": feature_id if feature_id else None,
            "platformId": used_platform_id,
            "timestamp": timestamp,
        }

        # Add metadata if provided
        if metadata:
            payload["metadata"] = metadata
        elif custom_deduction is not None:
            payload["metadata"] = {"customFeatureDeduction": custom_deduction}

        signature = cls._create_signature(payload)
        payload["signature"] = signature

        api_response = None
        result = None

        try:
            url = f"{cls.BASE_URL}/tokens/server/check-availability"
            cls.logger.info(f"Checking availability at URL: {url}")
            cls.logger.info(f"Payload being sent: {payload}")
            resp = requests.post(url, json=payload, headers=cls._headers(), timeout=10)
            cls.logger.info(f"Response status code: {resp.status_code}")
            cls.logger.info(f"Response content: {resp.content}")
            data = resp.json()
            cls.logger.info(f"Reservation check-availability response: {data}")

            # Store the full API response for logging
            api_response = {
                "status_code": resp.status_code,
                "response_data": data,
                "request_payload": payload,
            }

            if data.get("success", False):
                result = {"success": True, "data": data}
            else:
                result = {
                    "success": False,
                    "error": data.get("message", "Token check failed"),
                    "status_code": resp.status_code,
                    "data": data,
                }
        except requests.RequestException as e:
            cls.logger.error(
                f"Network error during availability check for user {user_id}: {str(e)}"
            )
            cls.logger.error(f"Request URL: {url}")
            cls.logger.error(f"Request payload: {payload}")
            cls.logger.error(f"Request headers: {cls._headers()}")
            import traceback

            cls.logger.error(f"Full traceback: {traceback.format_exc()}")
            api_response = {
                "error": f"Network error: {str(e)}",
                "status_code": 500,
                "request_payload": payload,
            }
            result = {
                "success": False,
                "error": f"Network error: {str(e)}",
                "status_code": 500,
            }
        except Exception as e:
            cls.logger.error(
                f"Unexpected error during availability check for user {user_id}: {str(e)}"
            )
            cls.logger.error(f"Request URL: {url}")
            cls.logger.error(f"Request payload: {payload}")
            import traceback

            cls.logger.error(f"Full traceback: {traceback.format_exc()}")
            api_response = {
                "error": f"Token validation error: {str(e)}",
                "status_code": 500,
                "request_payload": payload,
            }
            result = {
                "success": False,
                "error": f"Token validation error: {str(e)}",
                "status_code": 500,
            }

        # Log availability check event
        try:
            log_result = cls.log_generation(
                platform_id=used_platform_id,
                status="availability_check",
                user_id=str(user_id),
                email=email or "unknown",
            )
            if log_result and log_result.get("success"):
                cls.logger.info(f"✅ Availability check logged for user {user_id}")
            else:
                cls.logger.warning(f"Failed to log availability check for user {user_id}")
        except Exception as log_error:
            cls.logger.error(f"Error logging availability check: {log_error}")

        return result

    @classmethod
    def reserve_tokens(
        cls,
        user_id,
        feature_id=None,
        metadata=None,
        task_id=None,
        email=None,
        platform_id=None,
    ):
        """
        Reserve tokens for a long-running operation.
        Updated to match working test script pattern exactly.
        """
        cls.logger.info(
            f"Reserving tokens for user: {user_id}, feature_id: {feature_id}, metadata: {metadata}"
        )
        timestamp = int(time.time() * 1000)
        used_platform_id = platform_id if platform_id else cls.TOOL_CODE

        # Match exact payload structure from working test
        payload = {
            "userId": str(user_id),
            "featureId": feature_id if feature_id else None,
            "metadata": metadata
            or {
                "generationType": "content_generation",
                "requestId": f"req_{timestamp}",
                "testMode": False,
            },
            "platformId": used_platform_id,
            "timestamp": timestamp,
        }

        cls.logger.info(f"Reserve payload: {payload}")
        signature = cls._create_signature(payload)
        payload["signature"] = signature
        cls.logger.info(f"Reserve payload with signature: {payload}")

        api_response = None
        result = None
        reservation_id = None

        try:
            url = f"{cls.BASE_URL}/reservations/server/reserve"
            cls.logger.info(f"Reserving tokens at URL: {url}")
            resp = requests.post(url, json=payload, headers=cls._headers(), timeout=10)
            cls.logger.info(f"Reserve response status code: {resp.status_code}")
            cls.logger.info(f"Reserve response content: {resp.content}")
            print(resp.json())
            data = resp.json()
            cls.logger.info(f"Reservation reserve response: {data}")

            # Store the full API response for logging
            api_response = {
                "status_code": resp.status_code,
                "response_data": data,
                "request_payload": payload,
            }

            if data.get("success", False):
                # Handle nested data structure - API returns data in data.data
                response_data = data.get("data", {})
                # If the data is further nested, extract it
                if isinstance(response_data, dict) and "data" in response_data:
                    response_data = response_data.get("data", {})
                reservation_id = response_data.get("reservationId")
                reserved_tokens = response_data.get("tokensReserved")
                estimated_cost = response_data.get("estimatedCost")
                cls.logger.info(f"✅ Tokens reserved successfully for user {user_id}")
                cls.logger.info(f"   Reservation ID: {reservation_id}")
                cls.logger.info(f"   Reserved Tokens: {reserved_tokens}")
                cls.logger.info(f"   Estimated Cost: ${estimated_cost}")
                cls.logger.info(f"   Feature ID: {feature_id}")
                cls.logger.info(f"   Metadata: {metadata}")
                result = {
                    "success": True,
                    "data": data,
                    "reservation_id": reservation_id,
                }
            else:
                error_message = data.get("message", "Unknown error")
                cls.logger.error(f"❌ Token reservation failed for user {user_id}")
                cls.logger.error(f"   Error: {error_message}")
                cls.logger.error(f"   Status Code: {resp.status_code}")
                cls.logger.error(f"   Feature ID: {feature_id}")
                cls.logger.error(f"   Metadata: {metadata}")
                cls.logger.error(f"   Full Response: {data}")
                result = {
                    "success": False,
                    "error": error_message,
                    "status_code": resp.status_code,
                    "data": data,
                }
        except requests.RequestException as e:
            cls.logger.error(
                f"Network error during token reservation for user {user_id}: {str(e)}"
            )
            cls.logger.error(f"Reservation URL: {url}")
            cls.logger.error(f"Reservation payload: {payload}")
            cls.logger.error(f"Reservation headers: {cls._headers()}")
            import traceback

            cls.logger.error(f"Full traceback: {traceback.format_exc()}")
            api_response = {
                "error": f"Network error: {str(e)}",
                "status_code": 500,
                "request_payload": payload,
            }
            result = {
                "success": False,
                "error": f"Network error: {str(e)}",
                "status_code": 500,
            }
        except Exception as e:
            cls.logger.error(
                f"Unexpected error during token reservation for user {user_id}: {str(e)}"
            )
            cls.logger.error(f"Reservation URL: {url}")
            cls.logger.error(f"Reservation payload: {payload}")
            import traceback

            cls.logger.error(f"Full traceback: {traceback.format_exc()}")
            api_response = {
                "error": f"Reservation error: {str(e)}",
                "status_code": 500,
                "request_payload": payload,
            }
            result = {
                "success": False,
                "error": f"Reservation error: {str(e)}",
                "status_code": 500,
            }

        # Log generation event for successful token reservation
        if result and result.get("success"):
            try:
                # Log successful token reservation as a generation event
                log_result = cls.log_generation(
                    platform_id=used_platform_id,
                    status="reserved",  # Custom status for token reservation
                    user_id=str(user_id),
                    email=email or "unknown",
                )

                if log_result and log_result.get("success"):
                    cls.logger.info(
                        f"✅ Token reservation logged successfully for user {user_id}"
                    )
                else:
                    cls.logger.warning(
                        f"Failed to log token reservation event for user {user_id}"
                    )

            except Exception as log_error:
                cls.logger.error(f"Error logging token reservation event: {log_error}")

        return result

    @classmethod
    def get_reservation_status(
        cls, reservation_id, user_id=None, task_id=None, platform_id=None
    ):
        """
        Get current status of a reservation.
        Updated to match working test script pattern exactly.
        """
        cls.logger.info(
            f"Getting reservation status for reservation_id: {reservation_id}"
        )
        timestamp = int(time.time() * 1000)
        used_platform_id = platform_id if platform_id else cls.TOOL_CODE

        # Match exact payload structure from working test
        payload = {
            "platformId": used_platform_id,
            "timestamp": str(timestamp),  # String timestamp for GET requests
        }

        cls.logger.info(f"Status payload: {payload}")
        signature = cls._create_signature(payload)

        params = {
            "signature": signature,
            "timestamp": timestamp,
            "platformId": used_platform_id,
        }
        cls.logger.info(f"Status params: {params}")

        api_response = None
        result = None

        try:
            url = f"{cls.BASE_URL}/reservations/server/{reservation_id}/status"
            cls.logger.info(f"Getting status at URL: {url}")
            resp = requests.get(url, headers=cls._headers(), params=params, timeout=10)
            cls.logger.info(f"Status response status code: {resp.status_code}")
            cls.logger.info(f"Status response content: {resp.content}")
            data = resp.json()
            cls.logger.info(f"Reservation status response: {data}")

            # Store the full API response for logging
            api_response = {
                "status_code": resp.status_code,
                "response_data": data,
                "request_params": params,
                "request_payload": payload,
            }

            result = data
        except requests.RequestException as e:
            cls.logger.error(
                f"Network error getting reservation status for {reservation_id}: {str(e)}"
            )
            api_response = {
                "error": f"Network error: {str(e)}",
                "status_code": 500,
                "request_params": params,
                "request_payload": payload,
            }
            result = {
                "success": False,
                "error": f"Network error: {str(e)}",
                "status_code": 500,
            }
        except Exception as e:
            cls.logger.error(
                f"Unexpected error getting reservation status for {reservation_id}: {str(e)}"
            )
            api_response = {
                "error": f"Status error: {str(e)}",
                "status_code": 500,
                "request_params": params,
                "request_payload": payload,
            }
            result = {
                "success": False,
                "error": f"Status error: {str(e)}",
                "status_code": 500,
            }

        return result

    @classmethod
    def extend_reservation_timer(
        cls, reservation_id, user_id=None, task_id=None, platform_id=None
    ):
        """
        Extend reservation expiry time by 30 minutes from current time.
        Updated to match working test script pattern exactly.
        """
        cls.logger.info(
            f"Extending reservation timer for reservation_id: {reservation_id}"
        )
        timestamp = int(time.time() * 1000)
        used_platform_id = platform_id if platform_id else cls.TOOL_CODE

        # Match exact payload structure from working test
        payload = {"platformId": used_platform_id, "timestamp": timestamp}

        cls.logger.info(f"Extend timer payload: {payload}")
        signature = cls._create_signature(payload)
        payload["signature"] = signature
        cls.logger.info(f"Extend timer payload with signature: {payload}")

        api_response = None
        result = None

        try:
            url = f"{cls.BASE_URL}/reservations/server/{reservation_id}/extend-timer"
            cls.logger.info(f"Extending timer at URL: {url}")
            resp = requests.put(url, json=payload, headers=cls._headers(), timeout=10)
            cls.logger.info(f"Extend timer response status code: {resp.status_code}")
            cls.logger.info(f"Extend timer response content: {resp.content}")
            data = resp.json()
            cls.logger.info(f"Reservation extend-timer response: {data}")

            # Store the full API response for logging
            api_response = {
                "status_code": resp.status_code,
                "response_data": data,
                "request_payload": payload,
            }

            result = data
        except requests.RequestException as e:
            cls.logger.error(
                f"Network error extending timer for reservation {reservation_id}: {str(e)}"
            )
            api_response = {
                "error": f"Network error: {str(e)}",
                "status_code": 500,
                "request_payload": payload,
            }
            result = {
                "success": False,
                "error": f"Network error: {str(e)}",
                "status_code": 500,
            }
        except Exception as e:
            cls.logger.error(
                f"Unexpected error extending timer for reservation {reservation_id}: {str(e)}"
            )
            api_response = {
                "error": f"Extend timer error: {str(e)}",
                "status_code": 500,
                "request_payload": payload,
            }
            result = {
                "success": False,
                "error": f"Extend timer error: {str(e)}",
                "status_code": 500,
            }

        return result

    @classmethod
    def consume_reservation(
        cls,
        reservation_id,
        final_tokens_consumed=None,
        user_id=None,
        task_id=None,
        email=None,
        platform_id=None,
    ):
        """
        Finalize reservation and consume tokens.
        Updated to match working test script pattern exactly.
        """
        # Normalize token value: convert 15.0 to 15, but keep 3.5 as 3.5
        if final_tokens_consumed is not None:
            try:
                tokens_float = float(final_tokens_consumed)
                # If it's a whole number, convert to int for cleaner API payload
                if tokens_float == int(tokens_float):
                    final_tokens_consumed = int(tokens_float)
                else:
                    final_tokens_consumed = tokens_float
            except (ValueError, TypeError):
                pass  # Keep original value if conversion fails

        cls.logger.info(
            f"Consuming reservation for reservation_id: {reservation_id}, final_tokens_consumed: {final_tokens_consumed}"
        )
        timestamp = int(time.time() * 1000)
        used_platform_id = platform_id if platform_id else cls.TOOL_CODE

        # Match exact payload structure from working test
        payload = {
            "finalTokensConsumed": final_tokens_consumed,
            "platformId": used_platform_id,
            "timestamp": timestamp,
        }

        if final_tokens_consumed is not None:
            payload["finalTokensConsumed"] = final_tokens_consumed

        cls.logger.info(f"Consume payload: {payload}")
        signature = cls._create_signature(payload)
        payload["signature"] = signature
        cls.logger.info(f"Consume payload with signature: {payload}")

        api_response = None
        result = None

        try:
            url = f"{cls.BASE_URL}/reservations/server/{reservation_id}/consume"
            cls.logger.info(f"Consuming reservation at URL: {url}")
            resp = requests.put(url, json=payload, headers=cls._headers(), timeout=10)
            print(resp.json())
            cls.logger.info(f"Consume response status code: {resp.status_code}")
            cls.logger.info(f"Consume response content: {resp.content}")
            data = resp.json()
            cls.logger.info(f"Reservation consume response: {data}")

            # Enhanced logging for successful consumption
            if data.get("success", False):
                consumed_tokens = data.get("data", {}).get("tokensConsumed")
                actual_cost = data.get("data", {}).get("actualCost")
                cls.logger.info(
                    f"✅ Tokens consumed successfully for reservation {reservation_id}"
                )
                cls.logger.info(f"   Consumed Tokens: {consumed_tokens}")
                cls.logger.info(f"   Actual Cost: ${actual_cost}")
                cls.logger.info(f"   Final Tokens Requested: {final_tokens_consumed}")
            else:
                error_message = data.get("message", "Unknown error")
                cls.logger.error(
                    f"❌ Token consumption failed for reservation {reservation_id}"
                )
                cls.logger.error(f"   Error: {error_message}")
                cls.logger.error(f"   Status Code: {resp.status_code}")
                cls.logger.error(f"   Final Tokens Requested: {final_tokens_consumed}")
                cls.logger.error(f"   Full Response: {data}")

            # Store the full API response for logging
            api_response = {
                "status_code": resp.status_code,
                "response_data": data,
                "request_payload": payload,
            }

            result = data
        except requests.RequestException as e:
            cls.logger.error(
                f"Network error consuming reservation {reservation_id}: {str(e)}"
            )
            cls.logger.error(f"Consume URL: {url}")
            cls.logger.error(f"Consume payload: {payload}")
            cls.logger.error(f"Consume headers: {cls._headers()}")
            cls.logger.error(f"Final tokens consumed: {final_tokens_consumed}")
            import traceback

            cls.logger.error(f"Full traceback: {traceback.format_exc()}")
            api_response = {
                "error": f"Network error: {str(e)}",
                "status_code": 500,
                "request_payload": payload,
            }
            result = {
                "success": False,
                "error": f"Network error: {str(e)}",
                "status_code": 500,
            }
        except Exception as e:
            cls.logger.error(
                f"Unexpected error consuming reservation {reservation_id}: {str(e)}"
            )
            cls.logger.error(f"Consume URL: {url}")
            cls.logger.error(f"Consume payload: {payload}")
            cls.logger.error(f"Final tokens consumed: {final_tokens_consumed}")
            import traceback

            cls.logger.error(f"Full traceback: {traceback.format_exc()}")
            api_response = {
                "error": f"Consume error: {str(e)}",
                "status_code": 500,
                "request_payload": payload,
            }
            result = {
                "success": False,
                "error": f"Consume error: {str(e)}",
                "status_code": 500,
            }

        # Log generation event for successful token consumption
        if result and result.get("success"):
            try:
                # Log successful token consumption as a generation event
                log_result = cls.log_generation(
                    platform_id=used_platform_id,
                    status="consumed",  # Custom status for token consumption
                    user_id=str(user_id),
                    email=email or "unknown",
                )

                if log_result and log_result.get("success"):
                    cls.logger.info(
                        f"✅ Token consumption logged successfully for user {user_id}"
                    )
                else:
                    cls.logger.warning(
                        f"Failed to log token consumption event for user {user_id}"
                    )

            except Exception as log_error:
                cls.logger.error(f"Error logging token consumption event: {log_error}")

        return result

    @classmethod
    def release_reservation(
        cls, reservation_id, reason=None, user_id=None, task_id=None, platform_id=None
    ):
        """
        Release reservation and refund tokens.
        Updated to match working test script pattern exactly.
        """
        cls.logger.info(
            f"Releasing reservation for reservation_id: {reservation_id}, reason: {reason}"
        )
        timestamp = int(time.time() * 1000)

        used_platform_id = platform_id if platform_id else cls.TOOL_CODE
        # Match exact payload structure from working test
        payload = {"platformId": used_platform_id, "timestamp": timestamp}

        if reason:
            payload["reason"] = reason

        cls.logger.info(f"Release payload: {payload}")
        signature = cls._create_signature(payload)
        payload["signature"] = signature
        cls.logger.info(f"Release payload with signature: {payload}")

        api_response = None
        result = None

        try:
            url = f"{cls.BASE_URL}/reservations/server/{reservation_id}/release"
            cls.logger.info(f"Releasing reservation at URL: {url}")
            resp = requests.put(url, json=payload, headers=cls._headers(), timeout=10)
            cls.logger.info(f"Release response status code: {resp.status_code}")
            cls.logger.info(f"Release response content: {resp.content}")
            data = resp.json()
            cls.logger.info(f"Reservation release response: {data}")

            # Enhanced logging for successful release
            if data.get("success", False):
                refunded_tokens = data.get("data", {}).get("tokensRefunded")
                cls.logger.info(
                    f"✅ Tokens released successfully for reservation {reservation_id}"
                )
                cls.logger.info(f"   Refunded Tokens: {refunded_tokens}")
                cls.logger.info(f"   Release Reason: {reason}")
            else:
                error_message = data.get("message", "Unknown error")
                cls.logger.error(
                    f"❌ Token release failed for reservation {reservation_id}"
                )
                cls.logger.error(f"   Error: {error_message}")
                cls.logger.error(f"   Status Code: {resp.status_code}")
                cls.logger.error(f"   Release Reason: {reason}")
                cls.logger.error(f"   Full Response: {data}")

            # Store the full API response for logging
            api_response = {
                "status_code": resp.status_code,
                "response_data": data,
                "request_payload": payload,
            }

            result = data
        except requests.RequestException as e:
            cls.logger.error(
                f"Network error releasing reservation {reservation_id}: {str(e)}"
            )
            cls.logger.error(f"Release URL: {url}")
            cls.logger.error(f"Release payload: {payload}")
            cls.logger.error(f"Release headers: {cls._headers()}")
            cls.logger.error(f"Release reason: {reason}")
            import traceback

            cls.logger.error(f"Full traceback: {traceback.format_exc()}")
            api_response = {
                "error": f"Network error: {str(e)}",
                "status_code": 500,
                "request_payload": payload,
            }
            result = {
                "success": False,
                "error": f"Network error: {str(e)}",
                "status_code": 500,
            }
        except Exception as e:
            cls.logger.error(
                f"Unexpected error releasing reservation {reservation_id}: {str(e)}"
            )
            cls.logger.error(f"Release URL: {url}")
            cls.logger.error(f"Release payload: {payload}")
            cls.logger.error(f"Release reason: {reason}")
            import traceback

            cls.logger.error(f"Full traceback: {traceback.format_exc()}")
            api_response = {
                "error": f"Release error: {str(e)}",
                "status_code": 500,
                "request_payload": payload,
            }
            result = {
                "success": False,
                "error": f"Release error: {str(e)}",
                "status_code": 500,
            }

        # Log generation event for successful token release/refund
        if result and result.get("success"):
            try:
                # Log successful token release as a generation event
                log_result = cls.log_generation(
                    platform_id=used_platform_id,
                    status="refunded",  # Custom status for token refund
                    user_id=str(user_id),
                    email="unknown",
                )

                if log_result and log_result.get("success"):
                    cls.logger.info(
                        f"✅ Token refund logged successfully for user {user_id}"
                    )
                else:
                    cls.logger.warning(
                        f"Failed to log token refund event for user {user_id}"
                    )

            except Exception as log_error:
                cls.logger.error(f"Error logging token refund event: {log_error}")

        return result

    @staticmethod
    def extract_numeric_duration(video_length):
        """Extract numeric duration value from video_length string (in minutes)."""
        TokenReservationClient.logger.info(
            f"Extracting duration from video_length: {video_length}"
        )
        if not video_length:
            TokenReservationClient.logger.info(
                "No video length provided, using default 1.0 minutes"
            )
            return 1.0  # Default to 1 if no length provided

        # If already int or float, return as float
        if isinstance(video_length, (int, float)):
            TokenReservationClient.logger.info(
                f"Video length is already numeric: {video_length}"
            )
            return float(video_length)

        video_length_str = str(video_length).strip()
        video_length_lower = video_length_str.lower()
        TokenReservationClient.logger.info(
            f"Normalized video length: {video_length_lower}"
        )
        import re

        # e.g. "10", "10.5"
        match_numeric = re.fullmatch(r"(\d+(?:\.\d+)?)", video_length_str)
        if match_numeric:
            duration = float(match_numeric.group(1))
            TokenReservationClient.logger.info(
                f"Video length looks like a plain number: {duration}"
            )
            return duration

        # e.g. "1 hour", "10 min", "2.5 hrs"
        match = re.match(
            r"(\d+(?:\.\d+)?)\s*(hour|hr|hrs|minute|min)", video_length_lower
        )
        if match:
            duration = float(match.group(1))
            unit = match.group(2)
            TokenReservationClient.logger.info(f"Matched duration: {duration} {unit}")
            if unit in ["hour", "hr", "hrs"]:
                result = duration * 60
                TokenReservationClient.logger.info(
                    f"Converted hours to minutes: {result}"
                )
                return result
            elif unit in ["minute", "min"]:
                TokenReservationClient.logger.info(
                    f"Duration already in minutes: {duration}"
                )
                return duration
        if "hour" in video_length_lower or "hr" in video_length_lower:
            numbers = re.findall(r"\d+(?:\.\d+)?", video_length_lower)
            if numbers:
                duration = float(numbers[0])
                result = duration * 60
                TokenReservationClient.logger.info(
                    f"Found hour pattern, converted to minutes: {result}"
                )
                return result
        elif "minute" in video_length_lower or "min" in video_length_lower:
            numbers = re.findall(r"\d+(?:\.\d+)?", video_length_lower)
            if numbers:
                duration = float(numbers[0])
                TokenReservationClient.logger.info(f"Found minute pattern: {duration}")
                return duration
        TokenReservationClient.logger.info(
            "Using default fallback duration: 1.0 minutes"
        )
        return 1.0  # Final fallback if cannot parse

    @classmethod
    def check_availability_with_modules(
        cls,
        user_id,
        video_length,
        feature_id="script_generation",
        generation_type="content_generation",
        request_id=None,
        logger=None,
    ):
        """
        Check token availability for content generation with modules
        """
        cls.logger.info(
            f"Checking token availability for user: {user_id}, video_length: {video_length}, feature_id: {feature_id}, generation_type: {generation_type}, request_id: {request_id}"
        )
        cls.logger.info(f"Extracting video duration from: {video_length}")
        total_video_duration = cls.extract_numeric_duration(video_length)
        cls.logger.info(
            f"Calculated total video duration: {total_video_duration} minutes"
        )
        cls.logger.info(f"Checking token availability for user {user_id}")
        # Note: We don't pass task_id here because the TaskMetadata may not exist yet
        # at the availability check stage. The task_id will be logged in later stages
        # (reserve, consume, release) when the TaskMetadata definitely exists.
        check_result = cls.check_availability(
            user_id, feature_id=feature_id
        )

        if not check_result["success"]:
            cls.logger.error(
                f"Token availability check failed for user {user_id}: {check_result.get('error', cls.ERR_UNKNOWN)}"
            )
            return {
                "success": False,
                "error": cls.ERR_CHECK_AVAILABILITY,
                "details": check_result.get("error", cls.ERR_UNKNOWN),
            }

        check_data = check_result["data"]
        required_tokens = check_data.get("requiredTokens", 0)
        cls.logger.info(f"Required tokens: {required_tokens}")
        print(f"Required tokens: {required_tokens}")
        total_tokens_needed = required_tokens * total_video_duration
        available_tokens = check_data.get("availableTokens", 0)
        cls.logger.info(
            f"Required tokens: {required_tokens}, Total tokens needed: {total_tokens_needed}, Available tokens: {available_tokens}"
        )
        cls.logger.info(f"Availability check result: {check_result}")

        if available_tokens < total_tokens_needed:
            cls.logger.warning(
                f"Insufficient tokens for user {user_id} - needed: {total_tokens_needed}, available: {available_tokens}"
            )
            return {
                "success": False,
                "error": cls.ERR_INSUFFICIENT_TOKENS,
                "status": "insufficient_tokens",
                "required_tokens": total_tokens_needed,
                "available_tokens": available_tokens,
                "message": f"You need {total_tokens_needed} tokens to generate content for {video_length}, but you only have {available_tokens} tokens available.",
            }

        # if check_data.get('platform', {}).get('isFree', False):
        #     cls.logger.info(f"Platform is free for user {user_id}, no tokens needed")
        #     return {
        #         'success': True,
        #         'message': cls.ERR_FREE_TOOL,
        #         'tokens_reserved': 0
        #     }

        return {
            "success": True,
            "message": "Tokens available",
            "required_tokens": total_tokens_needed,
            "available_tokens": available_tokens,
            # 'message': f'You need {total_tokens_needed} tokens to generate content for {video_length}, but you only have {available_tokens} tokens available.'
        }

    # Example: High-level flow for content generation with reservation
    @classmethod
    def reserve_and_consume_for_content_generation(
        cls,
        user_id,
        video_length,
        feature_id="script_generation",
        generation_type="content_generation",
        request_id=None,
        logger=None,
        task_id=None,
        user_email=None,
    ):
        """
        Reserve tokens for content generation, and provide reservationId for later consumption.
        Returns reservation info or error.
        """
        cls.logger.info(
            f"Starting content generation reservation for user: {user_id}, video_length: {video_length}, feature_id: {feature_id}, generation_type: {generation_type}, request_id: {request_id}"
        )

        # 1. Check availability
        if logger:
            logger.info(
                f"Checking availability for user: {user_id}, feature_id: {feature_id}, generation_type: {generation_type}, request_id: {request_id}"
            )
        cls.logger.info(f"Extracting video duration from: {video_length}")
        total_video_duration = cls.extract_numeric_duration(video_length)
        cls.logger.info(
            f"Calculated total video duration: {total_video_duration} minutes"
        )

        cls.logger.info(f"Checking token availability for user {user_id}")
        check_result = cls.check_availability(
            user_id, feature_id=feature_id, task_id=task_id
        )
        cls.logger.info(f"Availability check result: {check_result}")

        if not check_result["success"]:
            cls.logger.error(
                f"Token availability check failed for user {user_id}: {check_result.get('error', cls.ERR_UNKNOWN)}"
            )
            return {
                "success": False,
                "error": cls.ERR_CHECK_AVAILABILITY,
                "details": check_result.get("error", cls.ERR_UNKNOWN),
            }

        check_data = check_result["data"]
        required_tokens = check_data.get("requiredTokens", 0)

        total_tokens_needed = float(required_tokens) * float(total_video_duration)
        print(
            f"tokens needed {total_tokens_needed} {total_video_duration} {video_length}"
        )
        available_tokens = check_data.get("availableTokens", 0)
        print(
            f"Required tokens: {required_tokens}, Total tokens needed: {total_tokens_needed}, Available tokens: {available_tokens}"
        )
        print(f"Check data: {check_data}")
        cls.logger.info(
            f"Token calculation - Required per minute: {required_tokens}, Video duration: {total_video_duration}, Total needed: {total_tokens_needed}, Available: {available_tokens}"
        )
        cls.logger.info(
            f"Total tokens needed: {total_tokens_needed}, Available tokens: {available_tokens}"
        )

        if available_tokens < total_tokens_needed or not check_data.get(
            "canAfford", False
        ):
            cls.logger.warning(f"❌ Insufficient tokens for user {user_id}")
            cls.logger.warning(f"   Required: {total_tokens_needed} tokens")
            cls.logger.warning(f"   Available: {available_tokens} tokens")
            cls.logger.warning(f"   Video Length: {video_length}")
            cls.logger.warning(f"   Feature ID: {feature_id}")
            cls.logger.warning(f"   Can Afford: {check_data.get('canAfford', False)}")
            return {
                "success": False,
                "error": cls.ERR_INSUFFICIENT_TOKENS,
                "status": "insufficient_tokens",
                "required_tokens": total_tokens_needed,
                "available_tokens": available_tokens,
                "message": f"You need {total_tokens_needed} tokens to generate content for {video_length}, but you only have {available_tokens} tokens available.",
            }

        if check_data.get("platform", {}).get("isFree", False):
            cls.logger.info(f"Platform is free for user {user_id}, no tokens needed")
            return {"success": True, "message": cls.ERR_FREE_TOOL, "tokens_reserved": 0}

        # 2. Reserve tokens
        # Normalize customFeatureDeduction: convert 15.0 to 15, but keep 3.5 as 3.5
        custom_deduction = total_tokens_needed
        try:
            deduction_float = float(custom_deduction)
            # If it's a whole number, convert to int for cleaner API payload
            if deduction_float == int(deduction_float):
                custom_deduction = int(deduction_float)
            else:
                custom_deduction = deduction_float
        except (ValueError, TypeError):
            pass  # Keep original value if conversion fails

        metadata = {
            "generationType": generation_type,
            "requestId": request_id or f"req_{int(time.time() * 1000)}",
            "customFeatureDeduction": (custom_deduction),
        }
        print(f"total_tokens_needed: {total_tokens_needed}")
        cls.logger.info(f"Reserving tokens with metadata: {metadata}")

        reserve_result = cls.reserve_tokens(
            user_id, feature_id=feature_id, metadata=metadata, task_id=task_id, email=user_email
        )
        cls.logger.info(f"Reservation result: {reserve_result}")
        print(f"Reservation reserve response: {reserve_result}")

        if not reserve_result["success"]:
            error_msg = reserve_result.get("error", cls.ERR_UNKNOWN)
            cls.logger.error(f"❌ Token reservation failed for user {user_id}")
            cls.logger.error(f"   Error: {error_msg}")
            cls.logger.error(f"   Feature ID: {feature_id}")
            cls.logger.error(f"   Video Length: {video_length}")
            cls.logger.error(f"   Generation Type: {generation_type}")
            cls.logger.error(f"   Request ID: {request_id}")
            cls.logger.error(f"   Full Reserve Result: {reserve_result}")
            print(f"Reservation reserve failed: {reserve_result}")
            return {
                "success": False,
                "error": cls.ERR_RESERVATION_FAILED,
                "details": error_msg,
                "data": reserve_result.get("data"),
            }

        reserve_data = reserve_result["data"]
        # Handle nested data structure: API returns {'success': True, 'data': {'reservationId': ...}}
        # So we need to get the inner 'data' field if it exists
        if isinstance(reserve_data, dict) and "data" in reserve_data:
            reserve_data = reserve_data["data"]

        # Get tokensReserved from API response (preserve decimal values like 1.25)
        tokens_reserved = reserve_data.get("tokensReserved")
        if tokens_reserved is not None:
            try:
                # Keep as float to preserve decimal values (e.g., 1.25)
                tokens_reserved = float(tokens_reserved)
            except (ValueError, TypeError):
                pass  # Keep original value if conversion fails

        cls.logger.info(
            f"✅ Content generation reservation successful for user {user_id}"
        )
        cls.logger.info(f"   Reservation ID: {reserve_data.get('reservationId')}")
        cls.logger.info(f"   Generation ID: {reserve_data.get('generationId')}")
        cls.logger.info(f"   Tokens Reserved: {tokens_reserved}")
        cls.logger.info(f"   Expires At: {reserve_data.get('expiresAt')}")
        cls.logger.info(f"   Feature ID: {feature_id}")
        cls.logger.info(f"   Video Length: {video_length}")
        cls.logger.info(f"   Request ID: {metadata['requestId']}")
        print(f"Reservation reserve data: {reserve_data}")

        return {
            "success": True,
            "message": cls.MSG_TOKENS_RESERVED,
            "reservation_id": reserve_data.get("reservationId"),
            "generation_id": reserve_data.get("generationId"),
            "tokens_reserved": tokens_reserved,
            "expires_at": reserve_data.get("expiresAt"),
            "platform": reserve_data.get("platform"),
            "feature_id": reserve_data.get("featureId"),
            "deduction_type": reserve_data.get("deductionType"),
            "custom_amount": reserve_data.get("customAmount"),
            "request_id": metadata["requestId"],
        }

    @classmethod
    def consume_reservation_and_finalize(
        cls, reservation_id, final_tokens_consumed=None, platform_id=None
    ):
        """
        Finalize reservation and consume tokens.
        """
        cls.logger.info(
            f"Finalizing reservation consumption for reservation_id: {reservation_id}, final_tokens_consumed: {final_tokens_consumed}"
        )
        consume_result = cls.consume_reservation(
            reservation_id,
            final_tokens_consumed=final_tokens_consumed,
            platform_id=platform_id,
        )
        cls.logger.info(f"Consume result: {consume_result}")

        if not consume_result.get("success", False):
            cls.logger.error(
                f"Failed to consume reservation {reservation_id}: {consume_result.get('error', cls.ERR_UNKNOWN)}"
            )
            return {
                "success": False,
                "error": cls.ERR_CONSUME_FAILED,
                "details": consume_result.get("error", cls.ERR_UNKNOWN),
                "data": consume_result,
            }

        cls.logger.info(f"Successfully consumed reservation {reservation_id}")
        return {
            "success": True,
            "message": cls.MSG_TOKENS_CONSUMED,
            "reservation_id": consume_result.get("reservationId"),
            "tokens_consumed": consume_result.get("tokensConsumed"),
            "tokens_unused": consume_result.get("tokensUnused"),
            "status": consume_result.get("status"),
        }

    @classmethod
    def release_reservation_and_refund(
        cls, reservation_id, reason=None, platform_id=None
    ):
        """
        Release reservation and refund tokens.
        """
        cls.logger.info(
            f"Releasing reservation and refunding tokens for reservation_id: {reservation_id}, reason: {reason}"
        )
        release_result = cls.release_reservation(
            reservation_id, reason=reason, platform_id=platform_id
        )
        cls.logger.info(f"Release result: {release_result}")

        if not release_result.get("success", False):
            cls.logger.error(
                f"Failed to release reservation {reservation_id}: {release_result.get('error', cls.ERR_UNKNOWN)}"
            )
            return {
                "success": False,
                "error": cls.ERR_RELEASE_FAILED,
                "details": release_result.get("error", cls.ERR_UNKNOWN),
                "data": release_result,
            }

        cls.logger.info(f"Successfully released reservation {reservation_id}")
        return {
            "success": True,
            "message": cls.MSG_TOKENS_RELEASED,
            "reservation_id": release_result.get("reservationId"),
            "tokens_refunded": release_result.get("tokensRefunded"),
            "status": release_result.get("status"),
        }

    @classmethod
    def log_generation(cls, platform_id, status, user_id, email):
        """
        Log generation event to the server.
        Python equivalent of the JavaScript logGeneration function.
        """
        cls.logger.info(
            f"Logging generation for platform: {platform_id}, status: {status}, user: {user_id}"
        )

        payload = {
            "platformId": platform_id,
            "status": status,
            "userId": user_id,
            "email": email,
            "timestamp": int(time.time() * 1000),
        }

        payload_string = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        signature = cls._create_signature(payload)

        try:
            url = f"{cls.BASE_URL}/usage/log-generation-server"
            cls.logger.info(f"Logging generation at URL: {url}")
            cls.logger.info(f"Log generation payload: {payload}")

            response_payload = {**payload, "signature": signature}

            resp = requests.post(
                url, json=response_payload, headers=cls._headers(), timeout=10
            )

            cls.logger.info(f"Log generation response status code: {resp.status_code}")
            cls.logger.info(f"Log generation response content: {resp.content}")

            if resp.status_code == 200:
                data = resp.json()
                cls.logger.info(f"✅ Generation logged successfully for user {user_id}")
                cls.logger.info(f"   Platform ID: {platform_id}")
                cls.logger.info(f"   Status: {status}")
                cls.logger.info(f"   Email: {email}")
                cls.logger.info(f"   Response: {data}")
                return data
            else:
                cls.logger.error(f"❌ Failed to log generation for user {user_id}")
                cls.logger.error(f"   HTTP Status: {resp.status_code}")
                cls.logger.error(f"   Response Text: {resp.text}")
                cls.logger.error(f"   Platform ID: {platform_id}")
                cls.logger.error(f"   Status: {status}")
                cls.logger.error(f"   Email: {email}")
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                    "status_code": resp.status_code,
                }

        except requests.RequestException as e:
            cls.logger.error(
                f"Network error logging generation for user {user_id}: {str(e)}"
            )
            cls.logger.error(f"Log generation URL: {url}")
            cls.logger.error(f"Log generation payload: {response_payload}")
            cls.logger.error(f"Log generation headers: {cls._headers()}")
            cls.logger.error(
                f"Platform ID: {platform_id}, Status: {status}, Email: {email}"
            )
            import traceback

            cls.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Network error: {str(e)}",
                "status_code": 500,
            }
        except Exception as e:
            cls.logger.error(
                f"Unexpected error logging generation for user {user_id}: {str(e)}"
            )
            cls.logger.error(f"Log generation URL: {url}")
            cls.logger.error(f"Log generation payload: {response_payload}")
            cls.logger.error(
                f"Platform ID: {platform_id}, Status: {status}, Email: {email}"
            )
            import traceback

            cls.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Logging error: {str(e)}",
                "status_code": 500,
            }

    @classmethod
    def deduct_tokens(
        cls,
        user_id,
        feature_id="sales-coaching",
        custom_deduction=1.0,
        email=None,
        platform_id=None,
    ):
        """
        Deduct tokens immediately using the reserve-and-consume pattern.
        Reserves tokens, then immediately consumes them.
        
        Args:
            user_id: The user ID to deduct tokens from
            feature_id: The feature ID for the deduction (default: sales-coaching)
            custom_deduction: Amount of tokens to deduct (default: 1.0)
            email: Optional user email for logging
            
        Returns:
            dict: Result with success status and details
        """
        cls.logger.info(f"Deducting {custom_deduction} tokens for user {user_id}, feature {feature_id}")
        used_platform_id = platform_id if platform_id else cls.TOOL_CODE
        
        # Step 1: Reserve tokens
        metadata = {
            "generationType": "token_deduction",
            "requestId": f"deduct_{int(time.time() * 1000)}",
            "customFeatureDeduction": custom_deduction,
        }
        
        reserve_result = cls.reserve_tokens(
            user_id=user_id,
            feature_id=feature_id,
            metadata=metadata,
            email=email,
            platform_id=used_platform_id,
        )
        
        if not reserve_result.get("success"):
            cls.logger.error(f"Token reservation failed for deduction: {reserve_result.get('error')}")
            return {
                "success": False,
                "error": reserve_result.get("error", "Reservation failed"),
                "stage": "reservation"
            }
        print(reserve_result)
        reservation_id = reserve_result.get("data")['reservationId']
        cls.logger.info(f"Tokens reserved for deduction: reservation_id={reservation_id}")
        
        # Validate reservation_id before consuming
        if not reservation_id or reservation_id == "None":
            cls.logger.error(f"Invalid reservation_id returned from reserve_tokens: {reservation_id}")
            return {
                "success": False,
                "error": f"Invalid reservation_id from API: {reservation_id}",
                "stage": "reservation",
                "reserve_response": reserve_result.get("data")
            }
        
        # Step 2: Immediately consume the reserved tokens
        consume_result = cls.consume_reservation_and_finalize(
            reservation_id=reservation_id,
            final_tokens_consumed=custom_deduction,
            platform_id=used_platform_id,
            # user_id=user_id,
            # email=email
        )
        
        if not consume_result.get("success"):
            cls.logger.error(f"Token consumption failed for deduction: {consume_result.get('error')}")
            # Try to release the reservation if consumption failed
            cls.release_reservation(
                reservation_id,
                reason="Consumption failed during deduct_tokens",
                platform_id=used_platform_id,
            )
            return {
                "success": False,
                "error": consume_result.get("error", "Consumption failed"),
                "stage": "consumption",
                "reservation_id": reservation_id
            }
        
        cls.logger.info(f"✅ Tokens deducted successfully: {custom_deduction} tokens for user {user_id}")
        return {
            "success": True,
            "message": "Tokens deducted successfully",
            "reservation_id": reservation_id,
            "tokens_consumed": consume_result.get("data", {}).get("tokensConsumed", custom_deduction),
            "actual_cost": consume_result.get("data", {}).get("actualCost")
        }

    @classmethod
    def log_generation_attempt(cls, user_id, email, platform_id, status):
        """
        Log a generation attempt to the server.
        Wrapper around log_generation for compatibility.
        """
        return cls.log_generation(
            platform_id=platform_id,
            status=status,
            user_id=str(user_id),
            email=email
        )

    # Additional helpers for batch and fixed operations can be implemented similarly,
    # using the reservation flow above.
