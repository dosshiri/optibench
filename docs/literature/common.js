// Claude のページとして開いたときは、リンク先を同じシリーズの Claude ページに切り替える（ファイルとして開いたときは相対パスのまま）
(function () {
  if (location.protocol === "file:" || /github\.io$|^localhost$|^127\./.test(location.hostname)) return;
  document.querySelectorAll("a[data-web]").forEach(function (a) {
    if (/^https:/.test(a.dataset.web)) { a.href = a.dataset.web; a.target = "_blank"; a.rel = "noopener"; }
  });
})();
