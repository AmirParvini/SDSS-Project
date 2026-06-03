<!DOCTYPE html>
<html lang="en">

<head>
    <meta lang="fa" charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <meta name="csrf-token" content="{{ csrf_token() }}">
    {{-- <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"> --}}
    @vite(['resources/css/app.css', 'resources/js/app.js'])
    @stack('styles')
    <title>@yield('title', 'SDSS')</title>
</head>

<body>
    <div class="container-fluid p-0 m-0 vh-100 d-flex flex-column overflow-hidden">
        <div class="position-relative shadow-sm w-100 rounded-b-xl bg-white" style="z-index: 2">
            <ul class="nav nav-underline" dir="rtl">
                <li class="nav-item pr-4">
                    <a class="nav-link active" aria-current="page" href="#">Dashboar</a>
                </li>
                <li class="nav-item pr-4">
                    <a class="nav-link" href="#">Data (HSC Parameters)</a>
                </li>
                <li class="nav-item pr-4">
                    <a class="nav-link" href="#">Reports</a>
                </li>
            </ul>
        </div>
        <div class="position-absolute dashboard container-fluid p-0 m-0 vh-100" style="z-index: 1">
            @yield('content')
        </div>
    </div>
</body>
{{-- <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script> --}}
@stack('scripts')
{{-- <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script> --}}

</html>
