import logo from './assets/logo.svg';
import search from './assets/search.svg';
import chat from './assets/chat-bubble.svg';
import './App.css';

function App() {
  return (
    <div className="App">
		<header className="App-header">
			<img src={logo} alt="logo"/>
			<p>Logout</p>
		</header>
      	<div className="App-body">
			<div className="App-left">
				<div className="search-create-div">
					<div className="search">
						<img src={search} alt="search"/>
						<input type="text" placeholder="Search" />
					</div>
					<div className="create">
						<img src={chat} alt="chat"/>
						Create Post
					</div>
				</div>
				<div className="post">
					<p>username</p>
					<h1>Title</h1>
				</div>
        	</div>
			<div className="App-right">
                <div className="post-comment-div">
                    <div className="post-body">
                        <p>username</p>
                        <h1>Title</h1>
                        <h2>Body</h2>
                    </div>
                    <div className="comment">
                        <p>username</p>
                        <h2>Comment</h2>
                    </div>
                </div>
                <div className="message">
                    <input type="text" placeholder="Comment Here" />
                </div>
			</div>
      	</div>
    </div>
  );
}

export default App;
