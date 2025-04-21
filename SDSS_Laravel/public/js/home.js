
// انتخاب آیتم‌ها و تغییر متن دکمه
document.querySelectorAll('.dropdown-item').forEach(item => {
    item.addEventListener('click', function() {
    const dropdownButton = this.closest('.btn-group').querySelector('.btn-secondary');
    dropdownButton.textContent = this.textContent;
    });
});


let controller;
let isFetching = false;
const report_id_array = ['Solution Status', 'Iterations', 'Solution Time (sec)', 'Total Distance', 'Total of Facalities Opened', 'Total Unmet Demand Amount'];

document.getElementById('run').addEventListener('click', function() {
    let APR = document.getElementById('APR').value;
    let PP = document.getElementById('PP').textContent;
    let Config = document.getElementById('Config').textContent;
    if (isFetching == false) {
        isFetching = true;
        this.innerHTML = 'stop';
        document.getElementById('error_message').innerHTML = '';
        controller = new AbortController();
        const signal = controller.signal;
        fetch('/solving', {
            signal,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').getAttribute('content')
            },
            body: JSON.stringify({
                'APR': APR,
                'PP': PP,
                'Config': Config,
            })
        })
        .then(response => {
            if (!response.ok) {
                // throw new Error('Network response was not ok');
                // console.log('error:', response);
            }
            return response.json();
        })
        .then(data => {
            if(data.status == 'success'){
                console.log(data)
                const keys = Object.keys(data.output);
                const values = Object.values(data.output);
                console.log("keys: ", keys);
                console.log("values: ", values);
                // for (let i = 0; i < keys.length; i++) {
                //     const report_row = document.getElementById(report_id_array[i]);
                //     report_row.innerHTML = '';
                //     report_row.innerHTML = values[i];
                // }
            }
            else if(data.status == 'error'){
                console.log('error:', data);
                let errorMsg = data.message || 'An error occurred';
                // if (data.error) {
                //     errorMsg += `: ${data.error}`;
                // }
                document.getElementById('error_message').innerHTML = errorMsg;
            }
            this.innerHTML = 'Solve Model';
            isFetching = false;
        })
        .catch(error => {
            if (error.name === 'AbortError') {
                document.getElementById('error_message').innerHTML = 'Fetch request was canceled.';
            } else {
                document.getElementById('error_message').innerHTML = 'Fetch failed.';
                console.error('Fetch error:', error);
            }
            this.innerHTML = 'Solve Model';
            isFetching = false;
        });
    }
    else{
        isFetching = false;
        controller.abort();
        this.innerHTML = 'Solve Model';
    }

});



var map = L.map('map').setView([35.7, 51.39], 11);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);


var blueCircle = L.divIcon({
    className: 'custom-icon',
    html: '<div style="width: 20px; height: 20px; background-color: blue; border-radius: 50%;"></div>'
});

var redSquare = L.divIcon({
    className: 'custom-icon', // کلاس CSS سفارشی
    html: '<div style="width: 20px; height: 20px; background-color: red;"></div>'
});