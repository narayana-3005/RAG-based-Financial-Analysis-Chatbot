// Import necessary libraries
const functions = require("firebase-functions"); // Firebase Cloud Functions SDK
const admin = require("firebase-admin"); // Firebase Admin SDK
const axios = require("axios"); // HTTP client for sending requests to Slack

// Initialize Firebase Admin SDK with default credentials
admin.initializeApp();

// Reference to Firebase Realtime Database
const db = admin.database();

// Set the threshold value for triggering the alert
const threshold = 100;

// Slack Incoming Webhook URL
const slackWebhookUrl = "https://hooks.slack.com/services/<application-slack-webhook_id>";

/**
 * Firebase Cloud Function to monitor changes to the 'incorrect_response_counter/count' node.
 * This function is triggered every time the value of '/incorrect_response_counter/count' is updated.
 */
exports.monitorNode = functions.database.ref("/incorrect_response_counter/count")
  .onUpdate((change, context) => {
    const value = change.after.val(); // Get the new value of 'count' after the update

    // Check if the value exceeds or reaches the threshold
    if (value >= threshold) {
      // If the threshold is crossed, send a Slack alert
      sendSlackAlert(value);
    }
  });

/**
 * Function to send a Slack alert when the threshold is exceeded.
 * This function sends a message to Slack through an incoming webhook.
 */
function sendSlackAlert(value) {
  // Create the message to send to Slack
  const message = {
    text: `🔔 Model re-training Alert: 'incorrect response clicks' crossed its weekly threshold! Current value: ${value}`,
    // Include the actual count value in the message (optional)
  };

  // Send the message to Slack using the webhook URL
  axios
    .post(slackWebhookUrl, message) // Make a POST request to the Slack webhook URL
    .then((response) => {
      console.log("Slack alert sent successfully:", response.data); // Log success response
    })
    .catch((error) => {
      console.error("Error sending Slack alert:", error); // Log any errors in sending the alert
    });
}
