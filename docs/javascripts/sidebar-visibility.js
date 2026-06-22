(function () {
  var path = window.location.pathname.replace(/\\/g, "/").toLowerCase();
  var keepSidebar = path.indexOf('/getting-started/') !== -1 || path.indexOf('/api/') !== -1;

  if (keepSidebar) {
    return;
  }

  document.documentElement.classList.add('hide-primary-sidebar');
})();
