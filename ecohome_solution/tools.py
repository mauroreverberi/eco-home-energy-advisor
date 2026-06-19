"""
Tools for EcoHome Energy Advisor Agent
"""
import os
import random
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from models.energy import DatabaseManager

# -- Changed -- (load .env so the tool also works when imported standalone, not only via a notebook)
load_dotenv()

# Initialize database manager
db_manager = DatabaseManager()

def _createOpenAiEmbeddings():
    return OpenAIEmbeddings(base_url="https://openai.vocareum.com/v1", api_key=os.getenv("VOCAREUM_API_KEY"))

@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast for a specific location and number of days.
    
    Args:
        location (str): Location to get weather for (e.g., "San Francisco, CA")
        days (int): Number of days to forecast (1-7)
    
    Returns:
        Dict[str, Any]: Weather forecast data including temperature, conditions, and solar irradiance
        E.g:
        forecast = {
            "location": ...,
            "forecast_days": ...,
            "current": {
                "temperature_c": ...,
                "condition": random.choice(["sunny", "partly_cloudy", "cloudy"]),
                "humidity": ...,
                "wind_speed": ...
            },
            "hourly": [
                {
                    "hour": ..., # for hour in range(24)
                    "temperature_c": ...,
                    "condition": ...,
                    "solar_irradiance": ...,
                    "humidity": ...,
                    "wind_speed": ...
                },
            ]
        }
    """

    days = max(1, min(days, 7))
    # seed the RNG with the location so the same place is reproducible
    rng = random.Random(location)

    loc = location.lower()

    if any(c in loc for c in ["phoenix", "miami", "houston", "dallas"]):
        base_temp = 30
    elif any(c in loc for c in ["seattle", "portland", "denver", "minneapolis"]):
        base_temp = 12
    else:
        base_temp = 20

    conditions = ["sunny", "partly_cloudy", "cloudy", "rainy"]
    weights = [0.4, 0.3, 0.2, 0.1]
    # how much sun reaches the panels per condition
    solar_factor = {"sunny": 1.0, "partly_cloudy": 0.6, "cloudy": 0.3, "rainy": 0.1}

    forecast = {
        "location": location,
        "forecast_days": days,
        "current": {
            "temperature_c": round(base_temp + rng.uniform(-3, 3), 1),
            "condition": rng.choices(conditions, weights=weights)[0],
            "humidity": rng.randint(35, 85),
            "wind_speed": round(rng.uniform(0, 25), 1),
        },
        "hourly": [],
    }

    for hour in range(24):
        condition = rng.choices(conditions, weights=weights)[0]
        daylight = max(0.0, 1 - abs(hour - 12) / 6)
        forecast["hourly"].append({
            "hour": hour,
            "temperature_c": round(base_temp + 5 * daylight + rng.uniform(-2, 2), 1),
            "condition": condition,
            "solar_irradiance": round(900 * daylight * solar_factor[condition], 1),
            "humidity": rng.randint(30, 90),
            "wind_speed": round(rng.uniform(0, 30), 1),
        })
    return forecast


@tool
def get_electricity_prices(date: str = None) -> Dict[str, Any]:
    """
    Get electricity prices for a specific date or current day.
    
    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)
    
    Returns:
        Dict[str, Any]: Electricity pricing data with hourly rates 
        E.g: 
        prices = {
            "date": ...,
            "pricing_type": "time_of_use",
            "currency": "USD",
            "unit": "per_kWh",
            "hourly_rates": [
                {
                    "hour": .., # for hour in range(24)
                    "rate": ..,
                    "period": ..,
                    "demand_charge": ...
                }
            ]
        }
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # Mock electricity pricing - in real implementation, this would call a pricing API
    # Use a base price per kWh
    # Then generate hourly rates with peak/off-peak pricing
    # Peak normally between 6 and 22...
    # demand_charge should be 0 if off-peak
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {"error": f"Invalid date format: '{date}'. Expected YYYY-MM-DD."}

    # seed by date so the same day is reproducible
    rng = random.Random(date)
    variation = rng.uniform(0.95, 1.05)

    # three time-of-use tiers; midday is cheaper, the evening is the real peak
    hourly_rates = []
    for hour in range(24):
        if hour < 6 or hour >= 22:
            period, base, demand_charge = "off_peak", 0.08, 0.0
        elif 16 <= hour < 21:
            period, base, demand_charge = "peak", 0.25, 0.05
        else:
            period, base, demand_charge = "mid_peak", 0.12, 0.02
        hourly_rates.append({
            "hour": hour,
            "rate": round(base * variation, 4),
            "period": period,
            "demand_charge": demand_charge,
        })

    prices = {
        "date": date,
        "pricing_type": "time_of_use",
        "currency": "USD",
        "unit": "per_kWh",
        "hourly_rates": hourly_rates,
        "summary": {
            "off_peak_rate": round(0.08 * variation, 4),
            "mid_peak_rate": round(0.12 * variation, 4),
            "peak_rate": round(0.25 * variation, 4),
            "off_peak_hours": "22:00-06:00",
            "mid_peak_hours": "06:00-16:00, 21:00-22:00",
            "peak_hours": "16:00-21:00",
        }
    }
    return prices

@tool
def query_energy_usage(start_date: str, end_date: str, device_type: str = None) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")
    
    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        records = db_manager.get_usage_by_date_range(start_dt, end_dt)

        if device_type:
            records = [r for r in records if r.device_type == device_type]

        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": []
        }

        for record in records:
            usage_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "consumption_kwh": record.consumption_kwh,
                "device_type": record.device_type,
                "device_name": record.device_name,
                "cost_usd": record.cost_usd
            })

        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}

@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        records = db_manager.get_generation_by_date_range(start_dt, end_dt)

        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(sum(r.generation_kwh for r in records) / max(1, (end_dt - start_dt).days), 2),
            "records": []
        }

        for record in records:
            generation_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "generation_kwh": record.generation_kwh,
                "weather_condition": record.weather_condition,
                "temperature_c": record.temperature_c,
                "solar_irradiance": record.solar_irradiance
            })

        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}

@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.
    
    Args:
        hours (int): Number of hours to look back (default 24)
    
    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)

        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(sum(r.consumption_kwh for r in usage_records), 2),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {}
            },
            "generation": {
                "total_generation_kwh": round(sum(r.generation_kwh for r in generation_records), 2),
                "average_weather": "sunny" if generation_records else "unknown"
            }
        }

        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += record.consumption_kwh
            summary["usage"]["device_breakdown"][device]["cost_usd"] += record.cost_usd or 0
            summary["usage"]["device_breakdown"][device]["records"] += 1

        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)

        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}

@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.
    
    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return
    
    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    try:
        # Initialize vector store if it doesn't exist
        persist_directory = "data/vectorstore"
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        # Load documents if vector store doesn't exist
        if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
            # Load documents
            documents = []
            # -- Changed -- (index ALL .txt docs in data/documents, not just the 2 starter files)
            doc_dir = "data/documents"
            for fname in sorted(os.listdir(doc_dir)):
                if not fname.endswith(".txt"):
                    continue
                doc_path = os.path.join(doc_dir, fname)
                loader = TextLoader(doc_path)
                docs = loader.load()
                documents.extend(docs)

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)

            # Create vector store
            embeddings = _createOpenAiEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        else:
            # Load existing vector store
            embeddings = _createOpenAiEmbeddings()
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )

        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=max_results)

        results = {
            "query": query,
            "total_results": len(docs),
            "tips": []
        }

        for i, doc in enumerate(docs):
            results["tips"].append({
                "rank": i + 1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "relevance_score": "high" if i < 2 else "medium" if i < 4 else "low"
            })

        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}

@tool
def calculate_energy_savings(device_type: str, current_usage_kwh: float,
                           optimized_usage_kwh: float, price_per_kwh: float = 0.12) -> Dict[str, Any]:
    """
    Calculate potential energy savings from optimization.
    
    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)
    
    Returns:
        Dict[str, Any]: Savings calculation results
    """
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0

    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(savings_usd * 365, 2)
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings
]
