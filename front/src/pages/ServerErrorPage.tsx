import "./ErrorPage.css";

function ServerErrorPage() {
    const retry = () => {
        window.location.assign("/");
    };

    return (
        <main className="error-page">
            <section className="error-page-card" role="alert" aria-live="assertive">
                <p className="error-page-code">500</p>
                <h1 className="error-page-title">Internal Server Error</h1>
                <p className="error-page-text">
                    Something went wrong on our side. Please try again in a moment.
                </p>
                <button className="error-page-action" type="button" onClick={retry}>
                    Try Again
                </button>
            </section>
        </main>
    );
}

export default ServerErrorPage;
