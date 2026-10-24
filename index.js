
let count = 0;

function increaseCount() {
    count++;
    document.getElementById("count").innerText = count;
}

function decreaseCount() {
    count--;
    document.getElementById("count").innerText = count;
}

// Intentional error: incorrect variable name
function resetCount() {
    counter = 0;
    document.getElementById("count").innerText = count;
}
      