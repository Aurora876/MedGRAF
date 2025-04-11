document.addEventListener("DOMContentLoaded", function () {
    // 获取 DOM 元素
    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-input");
    const totalIterationsSlider = document.getElementById("total-iterations");
    const testJSlider = document.getElementById("test-j");
    const saveEverySlider = document.getElementById("save-every");
    const loadingDiv = document.getElementById("loading");
    const imageContainer = document.getElementById("image-container");
    const progressBarFill = document.getElementById("progress-bar-fill");
    const generateButton = uploadForm.querySelector("button");

    let intervalId = null;
    let timeoutId = null;

    // 更新 Total Iterations 的显示值
    totalIterationsSlider.addEventListener('input', () => {
        const totalIterationsValue = parseInt(totalIterationsSlider.value, 10);
        document.getElementById('total-iterations-value').textContent = totalIterationsValue;

        // 更新 Save Every 的最大值和当前值（如果超过 Total Iterations）
        if (parseInt(saveEverySlider.value, 10) > totalIterationsValue) {
            saveEverySlider.value = totalIterationsValue;
            document.getElementById('save-every-value').textContent = totalIterationsValue;
        }
        saveEverySlider.max = totalIterationsValue;
    });

    // 更新 Save Every 的显示值
    saveEverySlider.addEventListener('input', () => {
        const saveEveryValue = parseInt(saveEverySlider.value, 10);
        document.getElementById('save-every-value').textContent = saveEveryValue;

        // 确保 Save Every 的值不超过 Total Iterations
        if (saveEveryValue > parseInt(totalIterationsSlider.value, 10)) {
            saveEverySlider.value = totalIterationsSlider.value;
            document.getElementById('save-every-value').textContent = totalIterationsSlider.value;
        }
    });

    // 更新 Test J 的显示值
    testJSlider.addEventListener('input', () => {
        const degreesMap = { 0: '0°', 18: '90°', 36: '180°', 54: '270°' };
        document.getElementById('test-j-value').textContent = degreesMap[testJSlider.value] || '0° (0)';
    });

    // 监听表单提交事件
    uploadForm.addEventListener("submit", function (event) {
        event.preventDefault(); // 阻止默认提交行为

        const file = fileInput.files[0];
        if (!file) {
            alert("Please select a file.");
            return;
        }

        const allowedTypes = ["image/png", "image/jpeg", "image/gif"];
        if (!allowedTypes.includes(file.type)) {
            alert("Invalid file type. Only PNG, JPEG, and GIF are allowed.");
            return;
        }

        // 获取用户输入的参数
        const totalIterations = parseInt(totalIterationsSlider.value, 10);
        const testJ = parseInt(testJSlider.value, 10);
        const saveEvery = parseInt(saveEverySlider.value, 10);

        // 禁用按钮
        generateButton.disabled = true;

        // 创建 FormData 对象
        const formData = new FormData();
        formData.append("file", file);
        formData.append("total_iterations", totalIterations);
        formData.append("test_j", testJ);
        formData.append("save_every", saveEvery);

        // 显示加载动画
        loadingDiv.style.display = "block";
        imageContainer.innerHTML = ""; // 清空图片容器

        let pollInterval = 1000; // 初始轮询间隔为 1 秒
        intervalId = setInterval(() => fetchProgress(pollInterval), pollInterval);

        // 发送请求到后端
        fetch("http://192.168.3.12:6006/generate", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                clearTimeout(timeoutId); // 清除超时计时器
                clearInterval(intervalId);
                loadingDiv.style.display = "none";

                if (data.error) {
                    throw new Error(data.error);
                }

                if (!data.images || !Array.isArray(data.images)) {
                    throw new Error("Invalid response format.");
                }

                // 遍历图片路径，动态创建 <img> 元素
                data.images.forEach((imageUrl) => {
                    const imgElement = document.createElement("img");
                    imgElement.src = imageUrl; // 设置图片路径
                    imgElement.alt = "Generated Image";
                    imgElement.onerror = () => {
                        imgElement.src = "https://via.placeholder.com/200"; // 占位符图片
                        imgElement.alt = "Image failed to load";
                    };
                    imageContainer.appendChild(imgElement);
                });
            })
            .catch((error) => {
                clearTimeout(timeoutId);
                clearInterval(intervalId);
                loadingDiv.style.display = "none";

                let errorMessage = "An unknown error occurred.";
                if (error.message.includes("timeout")) {
                    errorMessage = "The request timed out. Please try again.";
                } else if (error.message.includes("network")) {
                    errorMessage = "Network error. Please check your connection.";
                } else {
                    errorMessage = `Failed to load images: ${error.message}`;
                }

                imageContainer.innerHTML = `<p class="error-message">${errorMessage}</p>`;
            })
            .finally(() => {
                generateButton.disabled = false; // 重新启用按钮
            });
    });

    // 轮询进度
    function fetchProgress(currentInterval) {
        fetch("http://192.168.3.12:6006/progress")
            .then((response) => response.json())
            .then((data) => {
                const { current, total } = data;

                if (!current || !total) {
                    console.error("Invalid progress data:", data);
                    return;
                }

                const progressPercent = (current / total) * 100;
                progressBarFill.style.width = `${progressPercent}%`;
                progressBarFill.textContent = `${Math.round(progressPercent)}%`;

                // 根据进度调整颜色
                if (progressPercent < 30) {
                    progressBarFill.className = "progress-bar-fill high";
                } else if (progressPercent < 70) {
                    progressBarFill.className = "progress-bar-fill low";
                } else {
                    progressBarFill.className = "progress-bar-fill";
                }

                // 如果进度接近完成，减少轮询间隔
                if (progressPercent > 80 && currentInterval > 200) {
                    clearInterval(intervalId);
                    intervalId = setInterval(() => fetchProgress(200), 200); // 加快轮询
                }
            })
            .catch((error) => {
                console.error("Error fetching progress:", error);
            });
    }
});