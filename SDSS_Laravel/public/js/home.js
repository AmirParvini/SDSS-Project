// انتخاب آیتم‌ها و تغییر متن دکمه
document.querySelectorAll('.dropdown-item').forEach(item => {
    item.addEventListener('click', function() {
    const dropdownButton = this.closest('.btn-group').querySelector('.btn-secondary');
    dropdownButton.textContent = this.textContent;
    });
});

document.getElementById('myButton').addEventListener('click', function() {
    fetch('/solving', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').getAttribute('content')
        }
    })
    .then(response => response.json())
    .then(data => {
        // console.log('Success:', data);
        const keys = Object.keys(data);
        console.log('keys:', );
    })
    .catch(error => {
        console.error('Error:', error);
    });
});




var map = L.map('map').setView([35.7, 51.39], 11);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);