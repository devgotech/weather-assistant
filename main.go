package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"time"

	"regexp"

	"net/url"
	"os"
	"strings"

	"github.com/gage-technologies/mistral-go"
	"github.com/joho/godotenv"
	"golang.org/x/net/context"
)

// Load the API key from the .env file
func getAPIKey(envVar string) (string, error) {
	err := godotenv.Load()
	if err != nil {
		return "", fmt.Errorf("error loading .env file: %v", err)
	}
	apiKey := os.Getenv(envVar)
	if apiKey == "" {
		return "", fmt.Errorf("%s not set in .env file", envVar)
	}
	return apiKey, nil
}

// Extract the city name using Mistral
func extractCityFromUserInput(userMessage string) (string, error) {
	apiKey, err := getAPIKey("MISTRAL_API_KEY")
	if err != nil {
		return "", err
	}

	client := mistral.NewMistralClientDefault(apiKey)
	model := mistral.ModelOpenMistral7b

	//create a context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	//Simulate networ latency or clocking operation within the context
	done := make(chan struct{})
	var resp *mistral.ChatCompletionResponse

	go func() {
		// Ask Mistral to identify the city in the user's input
		messages := []mistral.ChatMessage{
			{
				Role:    mistral.RoleSystem,
				Content: "You are a weather assistant. Please extract only the city name in the following sentence and make sure the city is within quotes.",
			},
			{
				Role:    mistral.RoleUser,
				Content: userMessage,
			},
		}

		params := mistral.DefaultChatRequestParams
		// params.MaxTokens = 50
		// params.Temperature = 0

		resp, err = client.Chat(model, messages, &params)
		close(done)
	}()

	select {
	case <-ctx.Done():
		//handle context cancellation, e.g., timeout
		return "", fmt.Errorf("request timed out")
	case <-done:
		//proceed with processing the response
		if err != nil {
			return "", err
		}

		if len(resp.Choices) == 0 {
			return "", fmt.Errorf("no response choices from Mistral API")
		}
		// Extract and return the city name
		responseText := strings.TrimSpace(resp.Choices[0].Message.Content)
		re := regexp.MustCompile(`(?i)"([^"]+)"`) //matches text within quotes
		matches := re.FindStringSubmatch(responseText)
		if len(matches) < 2 {
			return "", fmt.Errorf("could not extract city name from Mistral's response")
		}

		city := matches[1]
		city = strings.TrimSpace(city)

		return city, nil
	}
}

// Fetch the weather data from OpenWeather API
func fetchWeatherData(city string) (map[string]interface{}, error) {
	apiKey, err := getAPIKey("WEATHER_API_KEY")
	if err != nil {
		return nil, err
	}

	// URL-encode the city name to ensure it is safe for inclusion in a URL
	encodedCity := url.QueryEscape(strings.TrimSpace(city))

	url := fmt.Sprintf("https://api.openweathermap.org/data/2.5/weather?q=%s&appid=%s&units=metric", encodedCity, apiKey)

	//log the API URL for debbuging
	log.Printf("Requesting weather data with URL: %s", url)

	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Check if the response status code is not 200 (OK)
	if resp.StatusCode != http.StatusOK {
		// Read the body in case of an error to get more details
		body, _ := ioutil.ReadAll(resp.Body)
		return nil, fmt.Errorf("failed to fetch weather data: status code %d, response: %s", resp.StatusCode, string(body))
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var weatherData map[string]interface{}
	err = json.Unmarshal(body, &weatherData)
	if err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %v, response body: %s", err, string(body))
	}

	return weatherData, nil
}

// Generate a response using Mistral with the weather data
func generateWeatherResponse(userMessage string, weatherData map[string]interface{}) (string, error) {
	apiKey, err := getAPIKey("MISTRAL_API_KEY")
	if err != nil {
		return "", err
	}

	client := mistral.NewMistralClientDefault(apiKey)
	model := mistral.ModelOpenMistral7b

	// Format the weather data into a string
	weatherInfo, err := formatWeatherResponse(weatherData)
	if err != nil {
		return "", err
	}

	//create a context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	//Simulate networ latency or clocking operation within the context
	done := make(chan struct{})
	var resp *mistral.ChatCompletionResponse

	go func() {
		// Pass the formatted weather information and user message to Mistral
		messages := []mistral.ChatMessage{
			{
				Role:    mistral.RoleSystem,
				Content: "You are a weather assistant. Use the following weather information to answer the user's question.",
			},
			{
				Role:    mistral.RoleSystem,
				Content: weatherInfo,
			},
			{
				Role:    mistral.RoleUser,
				Content: userMessage,
			},
		}

		params := mistral.DefaultChatRequestParams
		// params.MaxTokens = 50
		// params.Temperature = 0

		resp, err = client.Chat(model, messages, &params)
		close(done)
	}()

	select {
	case <-ctx.Done():
		//handle context cancellation, e.g., timeout
		return "", fmt.Errorf("request timed out")
	case <-done:
		//proceed with processing the response
		if err != nil {
			return "", err
		}

		if len(resp.Choices) == 0 {
			return "", fmt.Errorf("no response choices from Mistral API")
		}

		// Return the final response from Mistral
		responseMessage := strings.TrimSpace(resp.Choices[0].Message.Content)
		return responseMessage, nil
	}
}

// Format the weather data into a human-readable format
func formatWeatherResponse(data map[string]interface{}) (string, error) {
	// check if main exists and its a map
	mainData, ok := data["main"].(map[string]interface{})
	if !ok || mainData == nil {
		return "", fmt.Errorf("unexpected response format: 'main' key missing or invalid")
	}

	// check if the "weather" key exists and is a slice of interfaces
	weatherData, ok := data["weather"].([]interface{})
	if !ok || len(weatherData) == 0 {
		return "", fmt.Errorf("unexpected response format: 'weather' key missing or invalid")
	}

	// Get the first item from the "weather" slice and ensure it's a map
	weatherItem, ok := weatherData[0].(map[string]interface{})
	if !ok || weatherItem == nil {
		return "", fmt.Errorf("unexpected response format: 'weather[0]' item missing or invalid")
	}

	// Extract the fields safely
	temperature, tempOk := mainData["temp"].(float64)
	description, descOk := weatherItem["description"].(string)
	city, cityOk := data["name"].(string)

	// Ensure fields were extracted successfully
	if !tempOk || !descOk || !cityOk {
		return "", fmt.Errorf("unexpected response format: missing or invalid field(s)")
	}

	return fmt.Sprintf("The current weather in %s is %s with a temperature of %.2f℃.", city, description, temperature), nil
}

// Main function
func main() {
	fmt.Println("Ask about the weather")
	scanner := bufio.NewScanner(os.Stdin)
	if scanner.Scan() {
		userMessage := scanner.Text()

		// Step 1: Extract the city from the user's message
		city, err := extractCityFromUserInput(userMessage)
		if err != nil {
			fmt.Println("Error extracting city:", err)
			return
		}
		//ensure city is not empty
		if city == "" {
			fmt.Println("Could not extract city from your input")
			return
		}
		//log the extracted city name
		log.Printf("Extracted city: %s", city)

		// Step 2: Fetch the weather data for the extracted city
		weatherData, err := fetchWeatherData(city)
		if err != nil {
			log.Fatalf("Error fetching weather data: %v", err)
			return
		}

		// Step 3: Generate the final response using Mistral
		response, err := generateWeatherResponse(userMessage, weatherData)
		if err != nil {
			fmt.Println("Error generating response:", err)
			return
		}

		// Output the final response to the user
		fmt.Println(response)

	}

}
