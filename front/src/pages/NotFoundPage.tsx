import "./ErrorPage.css";

function NotFoundPage() {
    const goHome = () => {
        window.location.assign("/");
    };

    return (
        <main className="error-page">
            <section className="error-page-card" role="alert" aria-live="polite">
                <p className="error-page-code">404</p>
                <h1 className="error-page-title">Page Not Found</h1>
                <p className="error-page-text">
                    The page you requested doesn’t exist or may have been moved.
                </p>
                <button className="error-page-action" type="button" onClick={goHome}>
                    Go to Home
                </button>
            </section>
        </main>
    );
}

export default NotFoundPage;
