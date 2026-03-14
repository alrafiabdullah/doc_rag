import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";
import ServerErrorPage from "../pages/ServerErrorPage";

type Props = {
    children: ReactNode;
};

type State = {
    hasError: boolean;
};

class AppErrorBoundary extends Component<Props, State> {
    state: State = {
        hasError: false,
    };

    static getDerivedStateFromError(): State {
        return { hasError: true };
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        // console.error("Frontend runtime failure", error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return <ServerErrorPage />;
        }

        return this.props.children;
    }
}

export default AppErrorBoundary;
