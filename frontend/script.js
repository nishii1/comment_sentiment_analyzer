async function analyzeComments() {
    const videoUrl = document.getElementById("videoUrl").value;
    const errorMessage = document.getElementById("errorMessage");
    errorMessage.innerText = "";
  
    if (!videoUrl) {
      errorMessage.innerText = "Please enter a YouTube video URL.";
      return;
    }
  //5501
    try {
      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: videoUrl }),
      });
  
      if (!response.ok) throw new Error("Request failed");
  
      const result = await response.json();
      console.log(result);
      if (result.detail) {
        errorMessage.innerText = result.detail;
        return;
      }
  
      drawChart(result);
    } catch (err) {
      errorMessage.innerText = "Failed to fetch sentiment analysis.";
      console.error(err);
    }
  }
  
  
  function drawChart(data) {
    const ctx = document.getElementById("sentimentChart").getContext("2d");
  
    if (window.sentimentChart instanceof Chart) {
      window.sentimentChart.destroy();
    }
  
    window.sentimentChart = new Chart(ctx, {
      type: "pie",
      data: {
        labels: ["Positive", "Neutral", "Negative"],
        datasets: [{
          data: [data.positive, data.neutral, data.negative],
          backgroundColor: ["#4CAF50", "#FFC107", "#F44336"]
        }]
      },
      options: {
        // responsive: true,
        // maintainAspectRatio: false,
        responsive: true,
        plugins: {
          legend: {
            position: "bottom",
          },
          title: {
            display: true,
            text: "Sentiment Analysis Result"
          }
        }
      }
    });
    
    document.getElementById("chart-container").style.display = "block";
  }
  