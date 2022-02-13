import './App.css';
import Home from './pages/Home';
import CreatePost from './pages/CreatePost';
import SignIn from './pages/SignIn';
import {
  BrowserRouter as Router,
  Switch,
  Route
} from "react-router-dom";

function App() {
    return (
      <Router>
      <div>
        {/* <ul>
          <li>
            <Link to="/">Home</Link>
          </li>
          <li>
            <Link to="/create-post">Create Post</Link>
          </li>
          <li>
            <Link to="/dashboard">Dashboard</Link>
          </li>
        </ul>

        <hr /> */}

        {/*
          A <Switch> looks through all its children <Route>
          elements and renders the first one whose path
          matches the current URL. Use a <Switch> any time
          you have multiple routes, but you want only one
          of them to render at a time
        */}
        <Switch>
          <Route exact path="/">
            <Home />
          </Route>
          <Route path="/create-post">
            <CreatePost />
          </Route>
          <Route path="/signin">
            <SignIn />
          </Route>
        </Switch>
      </div>
    </Router>
    );
}

export default App;
